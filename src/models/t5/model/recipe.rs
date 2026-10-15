//! Editable, headless T5 encoder recipes for representation experiments.
//!
//! A block computes pre-normalized residual attention followed by its dense or
//! gated feed-forward branch. Attention is `softmax(Q K^T + relative_bias + mask) V`
//! with T5's scale of one: head count and width may change independently of
//! residual width `d_model`. Scores have `[batch, heads, query, key]` axes.
//!
//! Layers with equal head counts share one registered learned relative-bias owner.
//! Explicit token positions determine bucket lookups; padding constrains attention
//! even when a user supplies custom connectivity. Eager FP32 softmax exposes score
//! and pattern hooks, while the backend-default path retains its own arithmetic.
//! Staged execution preserves dtype, device, gradients, and a continuation cursor;
//! final normalization can be applied once to any retained encoder prefix.
//!
//! # Cost of one staged range
//!
//! A range reads the context's positions once (minimum, maximum, and whether
//! every row carries the same positions) in a single device-to-host transfer.
//! Each bias owner used by the range then builds its bias exactly once, on the
//! device, from that summary (see [`super::bias::positional_bias`]): a
//! `[1, heads, S, S]` plane when all rows share positions (the default
//! `0..S`), otherwise `[batch, heads, S, S]`. Blocks run through the same
//! [`T5Block::run`] graph as the complete model.

use std::collections::{BTreeMap, btree_map};

use burn::tensor::Int;

use super::*;
use crate::{
    experiment::AttentionPolicy, intervention::Stack, models::shared::checks::read_int_tensors,
};
use block::diagnostic_buffer;

// Per-block geometry and shared learned relative-position bias.

/// Per-layer choices in a T5 encoder recipe.
///
/// Attention projects `d_model` into `attention_heads * head_dim`, so changing
/// either value does not require changing the residual-stream width. Relative
/// position tables are shared by layers with the same attention head count.
#[derive(Clone, Debug, PartialEq)]
pub struct T5EncoderLayerSpec {
    /// Number of independently learned self-attention heads.
    pub attention_heads: usize,
    /// Width of each query, key, and value head.
    pub head_dim: usize,
    /// Intermediate width of the dense or gated feed-forward branch.
    pub feed_forward_width: usize,
    /// Feed-forward nonlinearity and dense versus gated topology.
    pub feed_forward_projection: T5FeedForwardProjection,
    /// Arithmetic path; eager FP32 softmax exposes attention scores and patterns.
    pub attention_precision: AttentionPrecisionMode,
}

impl T5EncoderLayerSpec {
    /// Copies the attention and feed-forward dimensions of one unmodified T5 encoder block.
    fn from_config(config: &T5Config) -> Self {
        Self {
            attention_heads: config.num_heads,
            head_dim: config.d_kv,
            feed_forward_width: config.d_ff,
            feed_forward_projection: config.feed_forward_proj,
            attention_precision: AttentionPrecisionMode::BackendDefault,
        }
    }

    /// Translates one edited block into a family configuration while retaining stack conventions.
    ///
    /// Clearing derived activation fields lets the explicit projection choice
    /// define dense versus gated topology without stale checkpoint aliases.
    fn config(&self, base: &T5Config) -> T5Config {
        let mut config = base.clone();
        config.num_layers = 1;
        config.num_decoder_layers = Some(1);
        config.num_heads = self.attention_heads;
        config.d_kv = self.head_dim;
        config.d_ff = self.feed_forward_width;
        config.feed_forward_proj = self.feed_forward_projection;
        config.dense_act_fn = None;
        config.is_gated_act = None;
        config
    }
}

/// An editable T5 encoder architecture, independent of any checkpoint weights.
///
/// Start from [`Self::from_config`], edit individual [`Self::layers`], and call
/// [`Self::init`] for random initialization. The recipe retains T5's unscaled
/// attention, pre-normalization, initializer variances, and relative bias
/// sharing. Layers with a new head count receive a new learned relative-bias
/// table; subsequent layers with that head count reuse it. An unchanged recipe
/// has the original single shared encoder bias table.
///
/// `config.num_layers` supplies the initial layer count. Afterwards the length
/// of `layers` is authoritative: truncate, remove, or append layer specifications
/// to change depth. Residual width, vocabulary, normalization, and relative-bias
/// bucket settings remain stack-wide choices in `config`.
#[derive(Clone, Debug, PartialEq)]
pub struct T5EncoderRecipe {
    /// Stack-wide T5 settings and defaults used to construct the initial layers.
    pub config: T5Config,
    /// Ordered per-layer architecture choices; its length determines depth.
    pub layers: Vec<T5EncoderLayerSpec>,
}

impl T5EncoderRecipe {
    /// Creates a faithful encoder recipe from a validated T5-family configuration.
    pub fn from_config(config: T5Config) -> Result<Self> {
        // Encoder-only: the causal decoder's stricter bucket bound does not apply.
        config.validate_stacks(T5Stacks::EncoderOnly)?;
        let layers = vec![T5EncoderLayerSpec::from_config(&config); config.num_layers];
        Ok(Self { config, layers })
    }

    /// Validates all layer dimensions before allocating any parameters.
    ///
    /// An empty layer list is allowed for embedding-plus-normalization controls.
    pub fn validate(&self) -> Result<()> {
        let mut base = self.config.clone();
        base.num_layers = 1;
        base.num_decoder_layers = Some(1);
        base.validate_stacks(T5Stacks::EncoderOnly)?;
        for (index, layer) in self.layers.iter().enumerate() {
            layer
                .config(&base)
                .validate_stacks(T5Stacks::EncoderOnly)
                .map_err(|error| {
                    Error::configuration(
                        format!("t5.encoder.recipe.layers.{index}"),
                        Error::from(error).to_string(),
                    )
                })?;
        }
        Ok(())
    }

    /// Initializes only the encoder and token embeddings, with random weights.
    ///
    /// No decoder or vocabulary projection is allocated. To reuse pretrained
    /// parameters, extract an existing model with
    /// [`T5ForConditionalGeneration::into_encoder`] or explicitly transfer
    /// selected parameter paths into this freshly initialized module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Result<T5Encoder<B>> {
        self.validate()?;
        let shared = EmbeddingConfig::new(self.config.vocab_size, self.config.d_model)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: self.config.initializer_factor,
            })
            .init(device);
        let mut owners = BTreeMap::new();
        let mut bias_sources = Vec::with_capacity(self.layers.len());
        let mut blocks = Vec::with_capacity(self.layers.len());
        for (index, specification) in self.layers.iter().enumerate() {
            let owner = *owners.entry(specification.attention_heads).or_insert(index);
            let mut block = T5Block::init(
                &specification.config(&self.config),
                false,
                owner == index,
                device,
            )?;
            block.self_attention.self_attention.attention.precision_mode =
                specification.attention_precision;
            blocks.push(block);
            bias_sources.push(owner);
        }
        Ok(T5Encoder {
            shared,
            encoder: T5Stack {
                block: blocks,
                final_layer_norm: t5_norm(&self.config, device)?,
                is_decoder: false,
            },
            recipe: self.clone(),
            bias_sources,
        })
    }
}

/// A T5 encoder backbone with editable layers and no language-model head.
///
/// Parameter paths intentionally match the source portions of
/// [`T5ForConditionalGeneration`]: `shared` and `encoder`. A classifier can pool
/// its hidden states and apply an ordinary Burn linear layer. This deterministic,
/// differentiable graph works with native Burn training. Add caller-owned
/// `Dropout` modules when needed; the complete upstream dropout topology is not
/// implemented here.
#[derive(Module, Debug)]
pub struct T5Encoder<B: Backend> {
    /// Token embeddings, preserving the full model's `shared` parameter path.
    pub shared: Embedding<B>,
    /// Transformer blocks and final normalization, without a decoder or LM head.
    pub encoder: T5Stack<B>,
    /// Architecture metadata, excluded from parameter traversal.
    #[module(skip)]
    pub recipe: T5EncoderRecipe,
    /// Learned relative-bias owner for each layer.
    #[module(skip)]
    bias_sources: Vec<usize>,
}

impl<B: Backend> T5ForConditionalGeneration<B> {
    /// Extracts the pretrained encoder and shared embeddings without reinitializing.
    ///
    /// The decoder and vocabulary head are dropped. Parameter identities,
    /// values, T5 bias sharing, and encoder numerical behavior are preserved.
    pub fn into_encoder(self) -> Result<T5Encoder<B>> {
        let recipe = T5EncoderRecipe::from_config(self.config)?;
        if self.encoder.is_decoder || self.encoder.block.len() != recipe.layers.len() {
            return Err(Error::configuration(
                "t5.encoder",
                "encoder topology disagrees with the source model configuration",
            ));
        }
        Ok(T5Encoder {
            shared: self.shared,
            bias_sources: vec![0; recipe.layers.len()],
            encoder: self.encoder,
            recipe,
        })
    }
}

impl<B: Backend> T5Encoder<B> {
    /// Lists the activation points supported by this encoder's actual layers.
    #[must_use]
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let mut points = vec![Stack::Encoder.embedding().info()];
        for (index, block) in self.encoder.block.iter().enumerate() {
            points.extend(block.activation_points(Layer::encoder(index)));
        }
        points.push(Stack::Encoder.final_norm().info());
        points
    }

    /// Embeds tokens into a raw residual stream, before any transformer layer.
    ///
    /// The returned context owns padding and explicit positions. Omitted positions
    /// are the original token indices `0..sequence`; supplying `position_ids`
    /// permits relative-position experiments. Edit `state.hidden` or its context
    /// before [`Self::run_layers`] to insert a module or change attention behavior.
    pub fn embed(&self, source: CausalLmInput<B>) -> Result<crate::experiment::HiddenState<B>> {
        use crate::experiment::{ExecutionContext, HiddenState};

        let model_device = self.shared.weight.lazy_device();
        // Relative-position experiments may supply explicit positions, which pass
        // through uninspected; tokens and mask rows share one readback.
        validate_input(
            &source,
            &model_device,
            &InputContract::new(ENCODER_INPUT, PARAMETERS, self.recipe.config.vocab_size)
                .with_implicit_positions(ImplicitPositions::Unused),
        )?;
        let [batch, sequence] = source.token_ids.dims();
        let positional_end = i64::try_from(sequence).map_err(|_| {
            Error::configuration(
                "t5.encoder.positions",
                "sequence length does not fit in i64",
            )
        })?;
        let positions = source.position_ids.unwrap_or_else(|| {
            Tensor::<B, 1, Int>::arange(0..positional_end, &model_device)
                .reshape([1, sequence])
                .expand([batch, sequence])
        });
        HiddenState::new(
            self.shared.forward(source.token_ids),
            ExecutionContext::new(positions, source.token_mask)?,
        )
    }

    /// Embeds tokens with capture or edits at the encoder embedding boundary.
    ///
    /// This scope contains only [`Stack::Encoder`]'s embedding point. Edits may
    /// call an external Burn module before the first transformer layer; positions
    /// and padding remain attached to the returned raw state.
    pub fn embed_with_hooks(
        &self,
        source: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<crate::experiment::HiddenState<B>> {
        hooks.run(vec![Stack::Encoder.embedding().info()], |hooks| {
            let mut state = self.embed(source)?;
            state.hidden = hooks.apply(Stack::Encoder.embedding(), state.hidden)?;
            Ok(state)
        })
    }

    /// Runs the next consecutive layer range without applying final normalization.
    ///
    /// `range.start` must equal `state.next_layer`. For example, run `0..2`,
    /// transform `state.hidden` with a small Burn module, then resume `2..4`.
    /// Attention policy and positions remain attached to the state across calls.
    pub fn run_layers(
        &self,
        state: crate::experiment::HiddenState<B>,
        range: std::ops::Range<usize>,
    ) -> Result<crate::experiment::HiddenState<B>> {
        self.run_layers_impl(state, range, &mut Hooks::new())
    }

    /// Runs a consecutive layer range with hooks scoped to those layers only.
    pub fn run_layers_with_hooks(
        &self,
        state: crate::experiment::HiddenState<B>,
        range: std::ops::Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<crate::experiment::HiddenState<B>> {
        // Build a bounded catalog without indexing an unvalidated user range.
        // Validation happens inside the scope so failures clear previous captures.
        let points = self
            .encoder
            .block
            .iter()
            .enumerate()
            .filter(|(index, _)| range.contains(index))
            .flat_map(|(index, block)| block.activation_points(Layer::encoder(index)))
            .collect();
        hooks.run(points, |hooks| self.run_layers_impl(state, range, hooks))
    }

    /// Applies the encoder's final normalization to a raw full or prefix output.
    ///
    /// This marks the result normalized; it cannot be resumed as a raw residual
    /// stream. The returned hidden tensor can be pooled and passed to any task head.
    pub fn normalize(
        &self,
        state: crate::experiment::HiddenState<B>,
    ) -> Result<crate::experiment::HiddenState<B>> {
        self.normalize_impl(state, &mut Hooks::new())
    }

    /// Applies final normalization with capture or edits at the final-norm point.
    pub fn normalize_with_hooks(
        &self,
        state: crate::experiment::HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<crate::experiment::HiddenState<B>> {
        hooks.run(vec![Stack::Encoder.final_norm().info()], |hooks| {
            self.normalize_impl(state, hooks)
        })
    }

    /// Encodes source tokens without creating vocabulary logits or a decoder.
    pub fn encode(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
    ) -> Result<EncoderOutput<B>> {
        self.encode_with_hooks(source, options, &mut Hooks::new())
    }

    /// Encodes source tokens with selective activation capture and intervention.
    ///
    /// The whole stack runs as one staged range, so positions are read once
    /// and each bias owner builds its bias once for every layer.
    pub fn encode_with_hooks(
        &self,
        source: CausalLmInput<B>,
        options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<EncoderOutput<B>> {
        hooks.run(self.activation_points(), |hooks| {
            validate_options(options)?;
            let mut state = self.embed(source)?;
            state.hidden = hooks.apply(Stack::Encoder.embedding(), state.hidden)?;
            let mut hidden_states =
                diagnostic_buffer(self.encoder.block.len(), options.hidden_states);
            if options.hidden_states {
                hidden_states.push(state.hidden.clone());
            }
            let retained = options.hidden_states.then_some(&mut hidden_states);
            state = self.run_layers_inner(state, 0..self.encoder.block.len(), hooks, retained)?;
            // The range just validated this state; only the final norm remains.
            state = self.apply_final_norm(state, hooks)?;
            if options.hidden_states {
                hidden_states.push(state.hidden.clone());
            }
            Ok(EncoderOutput {
                hidden_state: state.hidden,
                token_mask: state.context.token_mask,
                hidden_states,
                attentions: Vec::new(),
            })
        })
    }

    /// Checks `[batch, token, d_model]` residuals, dtype/device, cursor, and encoder-only policies.
    ///
    /// Reads the position summary (one device-to-host transfer) after every
    /// sync-free check, so invalid layouts fail without synchronizing.
    fn validate_state(&self, state: &crate::experiment::HiddenState<B>) -> Result<PositionSummary> {
        self.validate_state_layout(state)?;
        PositionSummary::read(state)
    }

    /// Performs every state check that needs no device values.
    fn validate_state_layout(&self, state: &crate::experiment::HiddenState<B>) -> Result<()> {
        let [batch, tokens, width] = state.hidden.dims();
        let d_model = self.recipe.config.d_model;
        if width != d_model || batch == 0 || tokens == 0 {
            return Err(Error::shape(
                "execution.hidden",
                format!("nonempty [batch, token, {d_model}]"),
                format!("{:?}", state.hidden.dims()),
            ));
        }
        state
            .context
            .validate_layout([batch, tokens], &state.hidden.device())?;
        ensure_same_device(
            "t5.encoder.hidden",
            &state.hidden.device(),
            &self.shared.weight.lazy_device(),
            PARAMETERS,
        )?;
        if state.hidden.dtype() != self.shared.weight.val().dtype() {
            return Err(Error::BackendCapability(
                "t5.encoder.hidden dtype must match encoder parameters".into(),
            ));
        }
        if self.encoder.is_decoder
            || state.context.encoder.is_some()
            || state.context.cross_attention.is_some()
            || !state.context.layer_cross_attention.is_empty()
        {
            return Err(Error::configuration(
                "t5.encoder.context",
                "an encoder has no decoder cross-attention context or policy",
            ));
        }
        if state.next_layer > self.encoder.block.len() {
            return Err(Error::configuration(
                "t5.encoder.next_layer",
                "continuation cursor exceeds encoder depth",
            ));
        }
        if state.context.layer_attention.iter().any(|(layer, _)| {
            layer.stack() != Stack::Encoder || layer.index() >= self.encoder.block.len()
        }) {
            return Err(Error::configuration(
                "t5.encoder.attention",
                "per-layer attention policy names a layer outside this encoder",
            ));
        }
        Ok(())
    }

    /// Executes a contiguous raw-residual range with a compatible learned bias owner per block.
    ///
    /// Each score bias has axes `[1 or batch, heads, query, key]` and derives
    /// from explicit positions. All attention policies are checked before
    /// execution; relative-bias lookups retain gradients to their single
    /// registered owner.
    fn run_layers_impl(
        &self,
        state: crate::experiment::HiddenState<B>,
        range: std::ops::Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<crate::experiment::HiddenState<B>> {
        self.run_layers_inner(state, range, hooks, None)
    }

    /// Runs `range` once validated, appending each block output to `retained` when given.
    ///
    /// Positions are read once for the whole range, and the bias of every
    /// owner the range uses is built once and shared by its layers.
    fn run_layers_inner(
        &self,
        mut state: crate::experiment::HiddenState<B>,
        range: std::ops::Range<usize>,
        hooks: &mut Hooks<'_, B>,
        mut retained: Option<&mut Vec<Tensor<B, 3>>>,
    ) -> Result<crate::experiment::HiddenState<B>> {
        let positions = self.validate_state(&state)?;
        state.validate_range(range.clone(), self.encoder.block.len())?;
        let [batch, sequence, _] = state.hidden.dims();
        for index in range.clone() {
            let policy = state.context.attention_for(Layer::encoder(index), false);
            policy.validate(
                [
                    batch,
                    self.encoder.block[index]
                        .self_attention
                        .self_attention
                        .attention
                        .query_heads,
                    sequence,
                    sequence,
                ],
                &state.hidden.device(),
            )?;
            if policy
                .bias
                .as_ref()
                .is_some_and(|bias| bias.dtype() != state.hidden.dtype())
            {
                return Err(Error::configuration(
                    "t5.encoder.attention.bias",
                    "bias dtype must match hidden states",
                ));
            }
        }
        let mut owner_biases: BTreeMap<usize, Tensor<B, 4>> = BTreeMap::new();
        for index in range.clone() {
            let owner = self.bias_owner(index)?;
            if let btree_map::Entry::Vacant(slot) = owner_biases.entry(owner) {
                slot.insert(positions.bias(self.owner_table(owner)?, &state)?);
            }
        }
        for index in range {
            let relative = owner_biases[&self.bias_owner(index)?].clone();
            let layer = Layer::encoder(index);
            let block = &self.encoder.block[index];
            let plan = policy_plan(
                state.context.attention_for(layer, false),
                state.context.token_mask.as_ref(),
                relative,
                [
                    batch,
                    block.self_attention.self_attention.attention.query_heads,
                    sequence,
                    sequence,
                ],
                &state.hidden.device(),
            )?;
            state.hidden =
                block.run(state.hidden, plan, &mut FreshKeyValues, None, layer, hooks)?;
            state.next_layer = index + 1;
            if let Some(retained) = retained.as_deref_mut() {
                retained.push(state.hidden.clone());
            }
        }
        Ok(state)
    }

    /// Returns the index of the block whose learned table biases layer `index`.
    fn bias_owner(&self, index: usize) -> Result<usize> {
        self.bias_sources.get(index).copied().ok_or_else(|| {
            Error::configuration(
                "t5.encoder.relative_bias",
                "layer list changed without updating the encoder recipe",
            )
        })
    }

    /// Returns the learned relative-position table registered at block `owner`.
    fn owner_table(&self, owner: usize) -> Result<&T5RelativePositionBias<B>> {
        self.encoder
            .block
            .get(owner)
            .and_then(|block| {
                block
                    .self_attention
                    .self_attention
                    .relative_attention_bias
                    .as_ref()
            })
            .ok_or_else(|| {
                Error::configuration(
                    "t5.encoder.relative_bias",
                    "missing compatible learned bias owner",
                )
            })
    }

    /// Applies final T5 RMS normalization once and preserves the result for an external task head.
    fn normalize_impl(
        &self,
        state: crate::experiment::HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<crate::experiment::HiddenState<B>> {
        self.validate_state(&state)?;
        self.apply_final_norm(state, hooks)
    }

    /// Normalizes a state whose layout and positions the caller already validated.
    fn apply_final_norm(
        &self,
        mut state: crate::experiment::HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<crate::experiment::HiddenState<B>> {
        if state.normalized {
            return Err(Error::configuration(
                "t5.encoder.normalize",
                "state is already normalized",
            ));
        }
        state.hidden = hooks.apply(
            Stack::Encoder.final_norm(),
            self.encoder.final_layer_norm.forward(state.hidden),
        )?;
        state.normalized = true;
        Ok(state)
    }
}

// Position validation and per-pass bias construction.

/// Host summary of a staged state's positions, read in one transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PositionSummary {
    /// Exclusive bound `E` with every position in `0..E` (the evaluated context).
    context: usize,
    /// Whether every batch row carries the same positions as row zero.
    uniform: bool,
}

impl PositionSummary {
    /// Checks positions against the evaluated context and detects batch-uniform rows.
    ///
    /// Reads `[min, max, rows_equal]` in one device-to-host transfer. The
    /// range check reproduces [`ExecutionContext::validate`](crate::experiment::ExecutionContext::validate),
    /// which a staged range would otherwise pay for with a separate read.
    fn read<B: Backend>(state: &crate::experiment::HiddenState<B>) -> Result<Self> {
        let positions = state.context.positions.clone();
        let [batch, sequence] = positions.dims();
        let first_row = positions.clone().narrow(0, 0, 1).expand([batch, sequence]);
        let uniform = positions.clone().equal(first_row).all().int();
        let values = read_int_tensors(
            vec![positions.clone().min(), positions.max(), uniform],
            "execution positions",
        )?;
        let (low, high) = (values[0], values[1]);
        let context = state.context.evaluated_context;
        if !(low >= 0 && usize::try_from(high).is_ok_and(|high| high < context)) {
            return Err(Error::configuration(
                "execution.positions",
                format!(
                    "positions must lie within the nonnegative evaluated context 0..{context}, found {low}..={high}"
                ),
            ));
        }
        Ok(Self {
            context,
            uniform: values[2] != 0,
        })
    }

    /// Builds one owner's bias from the validated positions.
    ///
    /// Positions lie in `0..E`, so every displacement is bounded by `E - 1`.
    /// Batch-uniform positions (the default `0..S`) produce a
    /// `[1, heads, S, S]` plane that broadcasts over the batch.
    fn bias<B: Backend>(
        self,
        table: &T5RelativePositionBias<B>,
        state: &crate::experiment::HiddenState<B>,
    ) -> Result<Tensor<B, 4>> {
        let positions = if self.uniform {
            state.context.positions.clone().narrow(0, 0, 1)
        } else {
            state.context.positions.clone()
        };
        bias::positional_bias(
            table,
            positions.clone(),
            positions,
            self.context.saturating_sub(1),
        )
    }
}

/// Combines an experiment attention policy with padding and the relative bias.
///
/// Padding always constrains attention (`true == valid` token mask). Without
/// a user connectivity plane the padding is passed through as a broadcast
/// `[batch, 1, 1, key]` view, so no `[batch, heads, query, key]` plane is
/// materialized; with one, [`AttentionPolicy::exclusion_mask`] unions them.
/// A policy bias is added to the relative bias.
fn policy_plan<B: Backend>(
    policy: AttentionPolicy<B>,
    token_mask: Option<&TokenMask<B>>,
    relative: Tensor<B, 4>,
    shape: [usize; 4],
    device: &B::Device,
) -> Result<SelfAttentionPlan<B>> {
    let padding = SelfAttentionPlan::padded(token_mask, None, policy.causal).excluded;
    let excluded = if policy.allowed.is_some() {
        let padding = padding.map(|plane| plane.expand(shape));
        policy.exclusion_mask(padding, shape, device)?
    } else {
        padding
    };
    let bias = match policy.expanded_bias(shape, device)? {
        Some(extra) => relative.expand(shape) + extra,
        None => relative,
    };
    Ok(SelfAttentionPlan {
        excluded,
        bias: Some(bias),
        causal: policy.causal,
    })
}

#[cfg(test)]
#[path = "recipe_tests.rs"]
mod tests;
