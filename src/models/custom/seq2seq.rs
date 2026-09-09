//! Modular encoder-decoder transformer assembly.
//!
//! Source blocks use bidirectional self-attention. Target blocks use causal self-attention,
//! then an independently parameterized cross-attention branch, then their configured dense or
//! routed feed-forward branch. Every cross-attention block uses the decoder layer's mixer-side
//! normalization convention because [`Seq2SeqSpec`] intentionally has no separate cross-norm
//! field.

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, LinearLayout},
    tensor::{ElementConversion, Tensor, backend::Backend},
};

use crate::{
    ArchitectureConfig, CausalLmInput, DecodeInput, EncoderOutput, Error, ForwardOptions, Result,
    Seq2SeqInput, Seq2SeqLanguageModel, Seq2SeqOutput, architecture::Seq2SeqSpec,
    cache::CacheConfig,
};

use super::{
    common::{
        AttentionDirection, BackboneOutput, ModularLayer, ModularNorm, causal_output,
        evaluated_context, has_moe, router_diagnostics, select_hidden, tied_projection,
        validate_and_resolve_input, validate_decoder_execution, validate_forward_options,
    },
    state::{ModularDecodeState, StatePurpose},
};

/// A modular bidirectional encoder and causal cross-attending decoder.
#[derive(Module, Debug)]
pub struct ModularSeq2SeqLm<B: Backend> {
    /// Source token embedding table.
    pub encoder_embed_tokens: Embedding<B>,
    /// Target token embedding table.
    pub decoder_embed_tokens: Embedding<B>,
    /// Bidirectional source layers.
    encoder_layers: Vec<ModularLayer<B>>,
    /// Causal target layers with cross-attention branches.
    decoder_layers: Vec<ModularLayer<B>>,
    /// Final source normalization.
    encoder_final_norm: ModularNorm<B>,
    /// Final target normalization.
    decoder_final_norm: ModularNorm<B>,
    /// Independent target vocabulary head, absent when tied to target embeddings.
    pub lm_head: Option<Linear<B>>,
    /// Validated topology excluded from parameter records.
    #[module(skip)]
    spec: Seq2SeqSpec,
}

impl<B: Backend> ModularSeq2SeqLm<B> {
    /// Validates both stacks and allocates the complete encoder-decoder parameter graph.
    pub fn init(spec: Seq2SeqSpec, device: &B::Device) -> Result<Self> {
        validate_seq2seq_execution(&spec)?;
        let encoder_layers = spec
            .encoder
            .layers
            .iter()
            .map(|layer| ModularLayer::init(layer, spec.encoder.hidden_size, None, device))
            .collect::<Result<Vec<_>>>()?;
        let decoder_layers = spec
            .decoder
            .layers
            .iter()
            .map(|layer| {
                ModularLayer::init(
                    layer,
                    spec.decoder.hidden_size,
                    Some((spec.cross_attention_heads, spec.cross_attention_head_dim)),
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            encoder_embed_tokens: EmbeddingConfig::new(
                spec.encoder.vocabulary_size,
                spec.encoder.hidden_size,
            )
            .init(device),
            decoder_embed_tokens: EmbeddingConfig::new(
                spec.decoder.vocabulary_size,
                spec.decoder.hidden_size,
            )
            .init(device),
            encoder_layers,
            decoder_layers,
            encoder_final_norm: ModularNorm::init(
                spec.encoder.final_norm,
                spec.encoder.hidden_size,
                device,
            )?,
            decoder_final_norm: ModularNorm::init(
                spec.decoder.final_norm,
                spec.decoder.hidden_size,
                device,
            )?,
            lm_head: (!spec.decoder.tie_word_embeddings).then(|| {
                crate::operations::init_linear(
                    &LinearConfig::new(spec.decoder.hidden_size, spec.decoder.vocabulary_size)
                        .with_bias(false)
                        .with_layout(LinearLayout::Col),
                    device,
                )
            }),
            spec,
        })
    }

    /// Returns the exact serializable topology used to construct both stacks.
    #[must_use]
    pub const fn spec(&self) -> &Seq2SeqSpec {
        &self.spec
    }

    /// Evaluates the bidirectional source stack.
    pub fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        validate_forward_options(options)?;
        if has_moe(&self.spec.encoder) && (options.router_logits || options.auxiliary_losses) {
            return Err(Error::BackendCapability(
                "EncoderOutput has no router-diagnostic field; encoder MoE diagnostics cannot be returned faithfully"
                    .into(),
            ));
        }
        let positions = validate_and_resolve_input(
            &source,
            self.spec.encoder.vocabulary_size,
            self.spec.encoder.max_position_embeddings,
            "custom.seq2seq.encoder",
        )?;
        let context = evaluated_context(positions.clone())?;
        let mut hidden = self.encoder_embed_tokens.forward(source.token_ids.clone());
        let mut hidden_states = Vec::with_capacity(if options.hidden_states {
            self.encoder_layers.len() + 2
        } else {
            0
        });
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        for layer in &self.encoder_layers {
            let (next, _router) = layer.forward(
                hidden,
                positions.clone(),
                source.token_mask.as_ref(),
                AttentionDirection::Bidirectional,
                context,
                None,
            )?;
            hidden = next;
            if options.hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.encoder_final_norm.forward(hidden);
        if options.hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(EncoderOutput {
            hidden_state: hidden,
            token_mask: source.token_mask,
            hidden_states,
            attentions: Vec::new(),
        })
    }

    /// Evaluates the causal target stack against an already encoded source.
    pub(super) fn forward_decoder_backbone(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: &CausalLmInput<B>,
        retain_hidden_states: bool,
    ) -> Result<BackboneOutput<B>> {
        validate_encoder_output(
            encoder,
            decoder.token_ids.dims()[0],
            self.spec.encoder.hidden_size,
            &decoder.token_ids.device(),
        )?;
        let positions = validate_and_resolve_input(
            decoder,
            self.spec.decoder.vocabulary_size,
            self.spec.decoder.max_position_embeddings,
            "custom.seq2seq.decoder",
        )?;
        let context = evaluated_context(positions.clone())?;
        let mut hidden = self.decoder_embed_tokens.forward(decoder.token_ids.clone());
        let mut hidden_states = Vec::with_capacity(if retain_hidden_states {
            self.decoder_layers.len() + 2
        } else {
            0
        });
        let mut routers = Vec::new();
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        for layer in &self.decoder_layers {
            let (next, router) = layer.forward(
                hidden,
                positions.clone(),
                decoder.token_mask.as_ref(),
                AttentionDirection::Causal,
                context,
                Some((&encoder.hidden_state, encoder.token_mask.as_ref())),
            )?;
            hidden = next;
            if let Some(router) = router {
                routers.push(router);
            }
            if retain_hidden_states {
                hidden_states.push(hidden.clone());
            }
        }
        hidden = self.decoder_final_norm.forward(hidden);
        if retain_hidden_states {
            hidden_states.push(hidden.clone());
        }
        Ok(BackboneOutput {
            hidden,
            hidden_states,
            routers,
        })
    }

    /// Applies an independent or target-embedding-tied vocabulary projection.
    pub(super) fn project_decoder(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        match &self.lm_head {
            Some(head) => crate::operations::linear(hidden, head),
            None => Ok(tied_projection(
                hidden,
                self.decoder_embed_tokens.weight.val(),
            )),
        }
    }

    /// Evaluates a complete stateless encoder-decoder graph.
    pub fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        validate_forward_options(options)?;
        let encoder = self.encode(input.encoder, options)?;
        let backbone =
            self.forward_decoder_backbone(&encoder, &input.decoder, options.hidden_states)?;
        let selected = select_hidden(
            backbone.hidden.clone(),
            input.decoder.token_mask.as_ref(),
            &options.logits,
        )?;
        let routers = router_diagnostics(
            &backbone.routers,
            input.decoder.token_mask.as_ref(),
            options,
        )?;
        let decoder = causal_output(
            self.project_decoder(selected)?,
            backbone.hidden,
            backbone.hidden_states,
            routers,
            options,
        );
        Ok(Seq2SeqOutput {
            decoder,
            encoder,
            cross_attentions: Vec::new(),
        })
    }
}

impl ArchitectureConfig for Seq2SeqSpec {
    const MODEL_TYPES: &'static [&'static str] = &["modular_seq2seq"];

    fn validate(&self) -> Result<()> {
        validate_seq2seq_execution(self)
    }
}

impl<B: Backend> Seq2SeqLanguageModel<B> for ModularSeq2SeqLm<B> {
    type Config = Seq2SeqSpec;
    type State = ModularDecodeState<B>;

    fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        ModularSeq2SeqLm::encode(self, source, options)
    }

    fn forward(
        &self,
        input: Seq2SeqInput<B>,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        ModularSeq2SeqLm::forward(self, input, options)
    }

    fn new_decode_state(
        &self,
        config: CacheConfig,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<Self::State> {
        if config.capacity() > self.spec.decoder.max_position_embeddings {
            return Err(Error::configuration(
                "custom.seq2seq.decode_state.capacity",
                format!(
                    "{} exceeds decoder max_position_embeddings {}",
                    config.capacity(),
                    self.spec.decoder.max_position_embeddings
                ),
            ));
        }
        ModularDecodeState::new(
            config,
            batch_size,
            self.spec.decoder.clone(),
            StatePurpose::Seq2Seq,
            device,
        )
    }

    fn prefill_decoder(
        &self,
        encoder: &EncoderOutput<B>,
        decoder: CausalLmInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        state.run_seq2seq(self, encoder, decoder, options, true)
    }

    fn decode(
        &self,
        encoder: &EncoderOutput<B>,
        input: DecodeInput<B>,
        state: &mut Self::State,
        options: &ForwardOptions,
    ) -> Result<Seq2SeqOutput<B>> {
        state.run_seq2seq(
            self,
            encoder,
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

/// Validates cross-stack topology and executable encoder/decoder capabilities.
fn validate_seq2seq_execution(spec: &Seq2SeqSpec) -> Result<()> {
    spec.validate()?;
    validate_decoder_execution(
        &spec.encoder,
        AttentionDirection::Bidirectional,
        "custom.seq2seq.encoder",
    )?;
    validate_decoder_execution(
        &spec.decoder,
        AttentionDirection::Causal,
        "custom.seq2seq.decoder",
    )
}

/// Checks externally supplied encoder states before cross-attention projection.
fn validate_encoder_output<B: Backend>(
    encoder: &EncoderOutput<B>,
    decoder_batch: usize,
    hidden_size: usize,
    decoder_device: &B::Device,
) -> Result<()> {
    let [batch, source_sequence, width] = encoder.hidden_state.dims();
    if batch != decoder_batch || source_sequence == 0 || width != hidden_size {
        return Err(Error::shape(
            "custom.seq2seq.encoder_output",
            format!("[{decoder_batch}, source>0, {hidden_size}]"),
            format!("[{batch}, {source_sequence}, {width}]"),
        ));
    }
    if encoder.hidden_state.device() != *decoder_device {
        return Err(Error::BackendCapability(
            "custom encoder output and decoder input must reside on the same device".into(),
        ));
    }
    if let Some(mask) = &encoder.token_mask
        && mask.as_tensor().dims() != [batch, source_sequence]
    {
        return Err(Error::shape(
            "custom.seq2seq.encoder_mask",
            format!("[{batch}, {source_sequence}]"),
            format!("{:?}", mask.as_tensor().dims()),
        ));
    }
    if let Some(mask) = &encoder.token_mask {
        if mask.as_tensor().device() != *decoder_device {
            return Err(Error::BackendCapability(
                "custom encoder mask and decoder input must reside on the same device".into(),
            ));
        }
        let minimum = mask
            .as_tensor()
            .clone()
            .int()
            .sum_dim(1)
            .min()
            .try_into_scalar()
            .map_err(|error| {
                Error::BackendCapability(format!("failed to validate custom encoder mask: {error}"))
            })?
            .elem::<i64>();
        if minimum == 0 {
            return Err(Error::shape(
                "custom.seq2seq.encoder_mask",
                "at least one valid source token in every row",
                "a fully masked row",
            ));
        }
    }
    Ok(())
}
