//! Select, observe, and change activations during one stateless forward pass.
//!
//! Start with a semantic point, rather than a checkpoint parameter name:
//! ```
//! use burn::tensor::backend::Backend;
//! use transformers::{CausalLmInput, ForwardOptions, intervention::{Hooks, Layer}};
//! use transformers::models::custom::ModularCausalLm;
//!
//! # fn inspect<B: Backend>(model: &ModularCausalLm<B>, input: CausalLmInput<B>)
//! # -> transformers::Result<()> {
//! let point = Layer::decoder(0).mlp_activation();
//! let mut hooks = Hooks::<B>::new();
//! let saved = hooks.capture(point.token(0));
//! hooks.edit(point.token(0), |x| x * 0.5);
//! let output = model.forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)?;
//! let original_first_token = hooks.get(&saved)?;
//! assert_eq!(original_first_token.dims()[1], 1);
//! # Ok(())
//! # }
//! ```
//!
//! Actions execute in registration order. Capture before an edit records the
//! original; capture after it records the edited value. Captures stay on device
//! and detach by default; [`Hooks::capture_with_grad`] explicitly retains the
//! gradient graph. Edits never detach implicitly. A new run clears old captures.
//! Observers ([`Hooks::observe`]) see a detached view and never rewrite the
//! activation, so they add no edit or constraint work to the forward pass.
//!
//! Registration mistakes (unknown points, impossible axes, empty or duplicate
//! selections) fail before any computation as [`Error::Configuration`].
//! Failures while the model runs (an out-of-range selection, a callback error,
//! a shape-changing edit, a violated constraint) are [`Error::Intervention`],
//! which names the point and keeps a callback's original error.
//!
//! Generic tools can turn a discovered [`PointInfo`] back into a typed point
//! with `Point::<D>::try_from(info)`, or capture any point regardless of rank
//! with [`Hooks::capture_dynamic`].
//!
//! # Derived points
//!
//! Attention with learned sinks drops its sink column after the softmax, so
//! an emitted pattern row sums to `1 - p_sink`. The derived, read-only point
//! [`AttentionPoints::sink_probability`] exposes that mass,
//! `p_sink[b, h, q] = 1 - sum_k pattern[b, h, q, k]`, through
//! [`Hooks::apply_sink_probability`]; it is computed only when requested, and
//! edits are rejected because changing it would not change the computation
//! (edit the pattern instead). Built-in attention does not emit this point
//! yet, so requesting it from a built-in model fails preflight as an unknown
//! point; custom attention emits it by calling
//! [`Hooks::apply_sink_probability`] right after emitting its pattern.
//!
//! # Mixture-of-experts points
//!
//! Sparse layers also emit router and expert-branch points, addressed like
//! every other layer point: [`Layer::router_logits`], [`Layer::router_weights`],
//! [`Layer::routed_output`], and [`Layer::shared_output`]. Router tensors end in
//! [`Axis::EXPERT`] or [`Axis::EXPERT_SLOT`], so
//! `Layer::decoder(3).router_logits().slice(Axis::EXPERT, 5..6)` selects one
//! expert's score. Each family emits the subset its routing defines, only in
//! sparse layers; `activation_points()` lists exactly what a model emits.
//!
//! # Custom models
//!
//! Closures may borrow ordinary Burn modules; their owner must register trainable
//! parameters in its `Module` tree. Hooks themselves are temporary execution state.
//! Custom models declare [`Point::custom`] and emit tensors with [`Hooks::apply`]
//! inside one [`Hooks::run`] scope. Built-in model wrappers create that scope for you.

use std::{
    ops::Range,
    sync::atomic::{AtomicU64, Ordering},
};

use burn::tensor::{Tensor, backend::Backend};

use crate::{Error, Result};

mod selection;
mod validity;
use selection::{
    ResolvedSlice, first_ill_formed_slice, resolve_slice, selection_ranges, unit_range,
};
pub use validity::PointValidity;

// Semantic tensor coordinates and rank-preserving selections.

/// Meaning of an activation dimension; rank alone does not identify its axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Independent examples.
    Batch,
    /// Sequence positions in a hidden state, projection, or head context.
    Token,
    /// Query or native key/value heads, as documented by the point.
    Head,
    /// Residual, neuron, or per-head channels, as documented by the point.
    Channel,
    /// Query positions of an attention score or probability matrix.
    Query,
    /// Key positions of an attention score or probability matrix.
    Key,
    /// An experiment-defined dimension without a built-in selection helper.
    Other(
        /// Stable experiment-defined label for this tensor dimension.
        &'static str,
    ),
}

impl Axis {
    /// Routed experts of a sparse mixture-of-experts layer, the last axis of
    /// [`Layer::router_logits`]; select experts with
    /// `point.slice(Axis::EXPERT, first..end)`.
    pub const EXPERT: Self = Self::Other("expert");
    /// Selected-expert slots of one token, the last axis of
    /// [`Layer::router_weights`]: slot `j` holds the `j`-th selected expert in
    /// decreasing router-score order.
    pub const EXPERT_SLOT: Self = Self::Other("expert_slot");
}

/// Discoverable description of a computation point, independent of tensor rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointInfo {
    /// Stable semantic name, such as `decoder.layers.0.mlp.activation`.
    pub name: String,
    /// Dimension meanings in their actual tensor order.
    pub axes: Vec<Axis>,
}

impl PointInfo {
    /// Tensor rank of activations emitted at this point.
    pub fn rank(&self) -> usize {
        self.axes.len()
    }
}

impl<const D: usize> TryFrom<PointInfo> for Point<D> {
    /// A configuration error naming the point when its catalog rank differs from `D`.
    type Error = Error;

    /// Recovers a typed point from a discovered catalog entry of rank `D`.
    ///
    /// ```
    /// use transformers::intervention::{Layer, Point, PointInfo};
    ///
    /// let info: PointInfo = Layer::decoder(0).mlp_activation().info();
    /// let point = Point::<3>::try_from(info.clone())?;
    /// assert_eq!(point, Layer::decoder(0).mlp_activation());
    /// assert!(Point::<4>::try_from(info).is_err());
    /// # Ok::<(), transformers::Error>(())
    /// ```
    fn try_from(info: PointInfo) -> Result<Self> {
        if info.rank() != D {
            return Err(Error::configuration(
                &info.name,
                format!(
                    "point has rank {} with axes {:?}; it cannot be used as a rank-{D} point",
                    info.rank(),
                    info.axes
                ),
            ));
        }
        Ok(Self { info })
    }
}

/// A computation point with a statically known rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Point<const D: usize> {
    /// Runtime point name and semantic axes, constrained to rank `D` at construction.
    info: PointInfo,
}

impl<const D: usize> Point<D> {
    /// Declare a point in a custom model and include it in that model's catalog.
    /// Names identify computation, not checkpoint keys. Axis declarations must
    /// match the tensor passed to [`Hooks::apply`].
    pub fn custom(name: impl Into<String>, axes: [Axis; D]) -> Self {
        Self {
            info: PointInfo {
                name: name.into(),
                axes: axes.into(),
            },
        }
    }

    /// Return the point's semantic name and axis layout for discovery.
    pub fn info(&self) -> PointInfo {
        self.info.clone()
    }

    /// Select one token while preserving its length-one dimension.
    pub fn token(&self, index: usize) -> Selection<D> {
        Selection::from(self).token(index)
    }

    /// Select one attention head while preserving its length-one dimension.
    pub fn head(&self, index: usize) -> Selection<D> {
        Selection::from(self).head(index)
    }

    /// Select one batch row while preserving its length-one dimension.
    pub fn batch(&self, index: usize) -> Selection<D> {
        Selection::from(self).batch(index)
    }

    /// Select a contiguous channel range.
    pub fn channels(&self, range: Range<usize>) -> Selection<D> {
        Selection::from(self).channels(range)
    }

    /// Select one query row of an attention matrix.
    pub fn query_token(&self, index: usize) -> Selection<D> {
        Selection::from(self).query_token(index)
    }

    /// Select one key column of an attention matrix.
    pub fn key_token(&self, index: usize) -> Selection<D> {
        Selection::from(self).key_token(index)
    }

    /// Select a contiguous range along a declared semantic axis.
    pub fn slice(&self, axis: Axis, range: Range<usize>) -> Selection<D> {
        Selection::from(self).slice(axis, range)
    }
}

/// A point and optional tensor slices; selections preserve rank and axis order.
#[derive(Clone, Debug)]
pub struct Selection<const D: usize> {
    /// Typed computation point selected by this operation.
    point: Point<D>,
    /// Requested semantic-axis ranges; unspecified axes retain their full extent.
    slices: Vec<(Axis, Range<usize>)>,
}

impl<const D: usize> From<Point<D>> for Selection<D> {
    /// Selects the complete typed point without changing its rank or axis order.
    fn from(point: Point<D>) -> Self {
        Self {
            point,
            slices: Vec::new(),
        }
    }
}

impl<const D: usize> From<&Point<D>> for Selection<D> {
    /// Clones a borrowed point into a complete, unsliced selection.
    fn from(point: &Point<D>) -> Self {
        Self::from(point.clone())
    }
}

impl<const D: usize> From<&Selection<D>> for Selection<D> {
    /// Clones a borrowed selection with its existing semantic-axis ranges.
    fn from(selection: &Selection<D>) -> Self {
        selection.clone()
    }
}

impl<const D: usize> Selection<D> {
    /// Select a range; nonexistent axes, duplicate axes, and invalid bounds are errors at execution.
    pub fn slice(mut self, axis: Axis, range: Range<usize>) -> Self {
        self.slices.push((axis, range));
        self
    }
    /// Select one token.
    pub fn token(self, index: usize) -> Self {
        self.one(Axis::Token, index)
    }
    /// Select one head.
    pub fn head(self, index: usize) -> Self {
        self.one(Axis::Head, index)
    }
    /// Select one independent example.
    pub fn batch(self, index: usize) -> Self {
        self.one(Axis::Batch, index)
    }
    /// Select neuron, residual, or per-head channels.
    pub fn channels(self, range: Range<usize>) -> Self {
        self.slice(Axis::Channel, range)
    }
    /// Select one query position in an attention matrix.
    pub fn query_token(self, index: usize) -> Self {
        self.one(Axis::Query, index)
    }
    /// Select one key position in an attention matrix.
    pub fn key_token(self, index: usize) -> Self {
        self.one(Axis::Key, index)
    }
    /// Selects a length-one semantic axis without overflowing the upper bound.
    ///
    /// At `usize::MAX` the range is empty and is rejected during scope
    /// validation rather than panicking during registration.
    fn one(self, axis: Axis, index: usize) -> Self {
        self.slice(axis, unit_range(index))
    }
}

/// Encoder or decoder stack; decoder-only models use [`Stack::Decoder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stack {
    /// Source encoder.
    Encoder,
    /// Target or decoder-only stack.
    Decoder,
}

impl Stack {
    /// Returns the stable stack prefix used by discoverable activation names.
    fn name(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::Decoder => "decoder",
        }
    }
    /// Embedded tokens entering the stack, `[batch, token, channel]`.
    pub fn embedding(self) -> Point<3> {
        hidden_point(format!("{}.embedding", self.name()))
    }
    /// Final normalized states, before any task head, `[batch, token, channel]`.
    pub fn final_norm(self) -> Point<3> {
        hidden_point(format!("{}.final_norm", self.name()))
    }

    /// Address layer `index` of this stack; `Stack::Decoder.layer(i) == Layer::decoder(i)`.
    pub const fn layer(self, index: usize) -> Layer {
        Layer::new(self, index)
    }
}

// Built-in layer and attention point catalogs.

/// Address a zero-based layer without knowing its checkpoint parameter names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layer {
    /// Encoder or decoder stack containing the layer.
    stack: Stack,
    /// Zero-based layer index within the stack.
    index: usize,
}

/// Generate documented hidden-state accessors with consistent tensor-axis semantics.
macro_rules! hidden_points {
    ($($method:ident => ($suffix:literal, $doc:literal)),* $(,)?) => {$ (
        /// The returned axes are `[batch, token, channel]`; token and channel selections preserve rank.
        #[doc = $doc]
        pub fn $method(self) -> Point<3> { hidden_point(format!("{}.layers.{}.{}", self.stack.name(), self.index, $suffix)) }
    )*};
}

impl Layer {
    /// Return the stack containing this layer.
    pub const fn stack(self) -> Stack {
        self.stack
    }

    /// Return this layer's zero-based index.
    pub const fn index(self) -> usize {
        self.index
    }

    /// Address layer `index` of `stack`, for code generic over encoder and decoder stacks.
    pub const fn new(stack: Stack, index: usize) -> Self {
        Self { stack, index }
    }

    /// Address an encoder layer.
    pub const fn encoder(index: usize) -> Self {
        Self {
            stack: Stack::Encoder,
            index,
        }
    }
    /// Address a decoder-only or target decoder layer.
    pub const fn decoder(index: usize) -> Self {
        Self {
            stack: Stack::Decoder,
            index,
        }
    }
    hidden_points! {
        residual_pre => ("residual.pre", "Residual entering this block, before its first normalization."),
        attention_input => ("attention.input", "Input to the configured self-attention or recurrent mixer, after any pre-sublayer normalization."),
        attention_output => ("attention.output", "Configured self-attention or recurrent mixer output, before residual addition."),
        cross_attention_input => ("cross_attention.input", "Query input to decoder cross attention, after any pre-sublayer normalization."),
        cross_attention_output => ("cross_attention.output", "Cross-attention output before its residual addition."),
        mlp_input => ("mlp.input", "Input to the feed-forward operation, after any pre-sublayer normalization."),
        mlp_pre_activation => ("mlp.pre_activation", "Dense/gate projection before its nonlinearity; gated MLPs expose the gate branch."),
        mlp_activation => ("mlp.activation", "After the nonlinearity, before output projection; gated MLPs expose the gate before multiplication."),
        mlp_output => ("mlp.output", "Feed-forward output before residual addition."),
        residual_post => ("residual.post", "Residual after the complete block, before the next block or final normalization."),
    }
    /// Input to this layer's sequence mixer; the same point as [`Self::attention_input`].
    ///
    /// Open compositions call this branch the *mixer* ([`crate::experiment::ResidualBlock`]'s
    /// `mixer` field); checkpoint families call it *attention* even for recurrent mixers
    /// such as DeltaNet. Both methods address the one emitted point
    /// `<stack>.layers.<index>.attention.input`, `[batch, token, channel]`.
    pub fn mixer_input(self) -> Point<3> {
        self.attention_input()
    }

    /// Output of this layer's sequence mixer; the same point as [`Self::attention_output`].
    ///
    /// Emitted before the residual addition as `<stack>.layers.<index>.attention.output`.
    pub fn mixer_output(self) -> Point<3> {
        self.attention_output()
    }

    // Sparse mixture-of-experts points. Only sparse layers emit them, and each
    // family emits the subset its routing defines; `activation_points()` lists
    // exactly what a model emits.

    /// Pre-softmax router scores of a sparse MoE layer, `[batch, token, expert]`.
    ///
    /// Emitted after the router projection and before expert selection by
    /// Qwen3-MoE, gpt-oss, and DeepSeek-V2/V3 (F32 there), so edits change which
    /// experts run and their mixture weights. The last axis is [`Axis::EXPERT`].
    /// Dense layers do not emit this point.
    pub fn router_logits(self) -> Point<3> {
        Point::custom(
            self.mlp_point_name("router_logits"),
            [Axis::Batch, Axis::Token, Axis::EXPERT],
        )
    }

    /// Mixture weights of the selected experts, `[batch, token, expert_slot]`.
    ///
    /// Emitted by Qwen3-MoE and gpt-oss after top-k selection and any
    /// renormalization: slot `j` weights the `j`-th selected expert in
    /// decreasing router-score order ([`Axis::EXPERT_SLOT`]). Edits rescale each
    /// selected expert's contribution without changing which experts run.
    pub fn router_weights(self) -> Point<3> {
        Point::custom(
            self.mlp_point_name("router_weights"),
            [Axis::Batch, Axis::Token, Axis::EXPERT_SLOT],
        )
    }

    /// Weighted sum of the routed experts, `[batch, token, channel]`.
    ///
    /// Emitted by DeepSeek-V2/V3 before the always-active shared experts are
    /// added, so edits reach only the routed branch.
    pub fn routed_output(self) -> Point<3> {
        hidden_point(self.mlp_point_name("routed_output"))
    }

    /// Output of the always-active shared experts, `[batch, token, channel]`.
    ///
    /// Emitted by DeepSeek-V2/V3 before it is added to [`Self::routed_output`].
    pub fn shared_output(self) -> Point<3> {
        hidden_point(self.mlp_point_name("shared_output"))
    }

    /// Names a point beside this layer's built-in MLP points: `<stack>.layers.<index>.mlp.<suffix>`.
    fn mlp_point_name(self, suffix: &str) -> String {
        format!("{}.layers.{}.mlp.{suffix}", self.stack.name(), self.index)
    }

    /// Address projected self-attention tensors in this layer.
    pub const fn self_attention(self) -> AttentionPoints {
        AttentionPoints {
            layer: self,
            cross: false,
        }
    }
    /// Address projected cross-attention tensors in this layer.
    pub const fn cross_attention(self) -> AttentionPoints {
        AttentionPoints {
            layer: self,
            cross: true,
        }
    }
}

/// Constructs a hidden-state point with `[batch, token, channel]` axis semantics.
fn hidden_point(name: String) -> Point<3> {
    Point::custom(name, [Axis::Batch, Axis::Token, Axis::Channel])
}

/// Typed attention points; only advertise points the executing model can expose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionPoints {
    /// Layer whose attention computation emits these activations.
    layer: Layer,
    /// Whether names address cross-attention instead of self-attention.
    cross: bool,
}

impl AttentionPoints {
    /// Normalized compressed key/value states before MLA expansion, `[batch, token, latent]`.
    pub fn kv_latent(self) -> Point<3> {
        hidden_point(format!(
            "{}.kv_latent",
            self.query().info().name.trim_end_matches(".query")
        ))
    }

    /// Normalized optional query bottleneck before MLA query expansion, `[batch, token, latent]`.
    pub fn query_latent(self) -> Point<3> {
        hidden_point(format!(
            "{}.query_latent",
            self.query().info().name.trim_end_matches(".query")
        ))
    }

    /// Builds a self/cross-attention point with explicit rank-four axis semantics.
    fn point(self, suffix: &str, axes: [Axis; 4]) -> Point<4> {
        Point::custom(
            format!(
                "{}.layers.{}.{}.{}",
                self.layer.stack.name(),
                self.layer.index,
                if self.cross {
                    "cross_attention"
                } else {
                    "attention"
                },
                suffix
            ),
            axes,
        )
    }
    /// Queries entering attention, after caller-applied normalization and position transforms.
    pub fn query(self) -> Point<4> {
        self.point(
            "query",
            [Axis::Batch, Axis::Head, Axis::Token, Axis::Channel],
        )
    }
    /// Keys entering attention after caller-applied transforms: native KV heads
    /// for grouped-query attention, expanded heads for multi-head latent attention.
    pub fn key(self) -> Point<4> {
        self.point("key", [Axis::Batch, Axis::Head, Axis::Token, Axis::Channel])
    }
    /// Values entering attention: native KV heads for grouped-query attention,
    /// expanded heads for multi-head latent attention.
    pub fn value(self) -> Point<4> {
        self.point(
            "value",
            [Axis::Batch, Axis::Head, Axis::Token, Axis::Channel],
        )
    }
    /// Scaled/softcapped scores, before additive bias and exclusion masks.
    pub fn scores(self) -> Point<4> {
        self.point("scores", [Axis::Batch, Axis::Head, Axis::Query, Axis::Key])
    }
    /// Attention probabilities after softmax and exclusions; see the executing method's edit constraints.
    pub fn pattern(self) -> Point<4> {
        self.point("pattern", [Axis::Batch, Axis::Head, Axis::Query, Axis::Key])
    }
    /// Probability mass each query row assigns to a learned attention sink, `[batch, head, query]`.
    ///
    /// Attention with sinks normalizes each row over the keys plus one sink
    /// column and then drops that column, so `p_sink = 1 - sum_k pattern`:
    /// zero when no mass reaches the sink and one for a row whose keys are
    /// all excluded. The point is derived and read-only; implementations emit
    /// it with [`Hooks::apply_sink_probability`] only where sinks exist and
    /// list it in their catalogs only then. Built-in attention (including
    /// modular specifications with `attention_sinks`) does not emit it yet, so
    /// only custom attention that calls `apply_sink_probability` provides it.
    pub fn sink_probability(self) -> Point<3> {
        Point::custom(
            format!(
                "{}.sink_probability",
                self.query().info().name.trim_end_matches(".query")
            ),
            [Axis::Batch, Axis::Head, Axis::Query],
        )
    }

    /// Per-query-head context before output gating and output projection.
    pub fn head_context(self) -> Point<4> {
        self.point(
            "head_context",
            [Axis::Batch, Axis::Head, Axis::Token, Axis::Channel],
        )
    }
    /// Catalog the six attention points, without claiming a model supports them.
    ///
    /// The derived [`Self::sink_probability`] point is not included; models
    /// with attention sinks list it separately.
    pub fn all(self) -> Vec<PointInfo> {
        vec![
            self.query().info(),
            self.key().info(),
            self.value().info(),
            self.scores().info(),
            self.pattern().info(),
            self.head_context().info(),
        ]
    }
}

/// Typed handle to a selected capture, owned by the [`Hooks`] that created it.
#[derive(Clone, Debug)]
pub struct Capture<const D: usize> {
    /// Unique hook-collection identity that prevents cross-collection handle reuse.
    owner: u64,
    /// Index of the capture action in its owning collection.
    entry: usize,
}

/// Handle to a capture whose rank is known only at run time.
///
/// Created by [`Hooks::capture_dynamic`] for catalog-driven tools that capture
/// every point a model lists without knowing ranks at compile time.
#[derive(Clone, Debug)]
pub struct DynamicCapture {
    /// Unique hook-collection identity that prevents cross-collection handle reuse.
    owner: u64,
    /// Index of the capture action in its owning collection.
    entry: usize,
    /// Captured point, retained so results can be labeled.
    point: PointInfo,
}

impl DynamicCapture {
    /// The captured point's name and axes.
    pub fn point(&self) -> &PointInfo {
        &self.point
    }
}

/// A captured activation of run-time rank: row-major values plus their shape.
#[derive(Clone, Debug)]
pub struct DynamicActivation<B: Backend> {
    /// Flattened row-major values, on the device that produced them.
    pub values: Tensor<B, 1>,
    /// Original dimensions; `values` holds their product of elements.
    pub shape: Vec<usize>,
}

impl<B: Backend> DynamicActivation<B> {
    /// Restores the original rank-`D` tensor; fails if the activation has another rank.
    pub fn into_tensor<const D: usize>(self) -> Result<Tensor<B, D>> {
        let shape: [usize; D] = self.shape.clone().try_into().map_err(|_| {
            Error::shape(
                "hooks.dynamic_activation",
                format!("rank {D}"),
                format!("{:?}", self.shape),
            )
        })?;
        Ok(self.values.reshape(shape))
    }
}

/// Rank-independent captured activation retained with its original shape.
struct Saved<B: Backend> {
    /// Flattened activation, detached only when requested by the capture policy.
    tensor: Tensor<B, 1>,
    /// Runtime dimensions restored when reading or editing the capture.
    shape: Vec<usize>,
}
/// Borrowed mutable callback receiving a flattened activation and its runtime shape.
type Edit<'a, B> = Box<dyn FnMut(Tensor<B, 1>, Vec<usize>) -> Result<Saved<B>> + 'a>;
/// Borrowed observer receiving a detached, flattened selection and its runtime shape.
type Observer<'a, B> = Box<dyn FnMut(Tensor<B, 1>, Vec<usize>) -> Result<()> + 'a>;
/// One ordered observer or mutation registered at a computation point.
enum Action<'a, B: Backend> {
    /// Retain the selected activation when its point is emitted.
    Capture {
        /// Whether the saved tensor must start outside the active gradient graph.
        detach: bool,
        /// Most recent successful capture, cleared at each run boundary or failure.
        saved: Option<Saved<B>>,
    },
    /// Replace a selection through a shape-checked callback.
    Edit(
        /// Rank-erased edit callback invoked in registration order.
        Edit<'a, B>,
    ),
    /// Inspect a detached selection without changing the activation.
    Observe(
        /// Rank-erased observer invoked in registration order.
        Observer<'a, B>,
    ),
}
/// Registered point selection and action together with run-local visit accounting.
struct Entry<'a, B: Backend> {
    /// Point name and semantic axes used during preflight and execution.
    point: PointInfo,
    /// Optional rank-preserving ranges to capture or replace, by semantic axis.
    slices: Vec<(Axis, Range<usize>)>,
    /// The same ranges resolved to tensor dimensions for the verified selection core.
    resolved: Vec<ResolvedSlice>,
    /// Capture, observer, or edit to perform when the selected point is reached.
    action: Action<'a, B>,
    /// Number of emissions seen during the current scoped execution.
    visits: usize,
}

impl<'a, B: Backend> Entry<'a, B> {
    /// Registers an action, resolving each semantic axis to its tensor dimension once.
    ///
    /// The verified [`resolve_slice`] picks the first dimension declaring the
    /// axis and counts every declaration. An axis absent from the point resolves
    /// to zero occurrences and is rejected by preflight; its placeholder
    /// dimension is never used.
    fn new(point: PointInfo, slices: Vec<(Axis, Range<usize>)>, action: Action<'a, B>) -> Self {
        let resolved = slices
            .iter()
            .map(|(axis, range)| {
                let matches = point
                    .axes
                    .iter()
                    .map(|candidate| candidate == axis)
                    .collect::<Vec<_>>();
                resolve_slice(&matches, range.clone())
            })
            .collect();
        Self {
            point,
            slices,
            resolved,
            action,
            visits: 0,
        }
    }

    /// Rejects ambiguous, missing, duplicate, or empty axis selections before execution.
    fn validate_selection(&self) -> Result<()> {
        let Some(index) = first_ill_formed_slice(&self.resolved) else {
            return Ok(());
        };
        let (axis, range) = &self.slices[index];
        let reason = if self.resolved[index].occurrences != 1 {
            format!(
                "selection needs exactly one {axis:?} axis; found {:?}",
                self.point.axes
            )
        } else if range.start >= range.end {
            format!("empty {axis:?} selection {range:?}")
        } else {
            format!("duplicate {axis:?} selection; select each axis at most once")
        };
        Err(Error::configuration(&self.point.name, reason))
    }
}
/// Next hook-collection identity used to reject captures belonging to another owner.
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

// Run-scoped action registration, capture ownership, and execution.

/// Scoped, ordered activation actions. Create one collection per experiment run policy.
/// No `Send` or `'static` bound is imposed on callbacks, so local modules can be borrowed.
pub struct Hooks<'a, B: Backend> {
    /// Collection identity attached to all capture handles issued here.
    owner: u64,
    /// Registered actions in the exact order that execution must apply them.
    entries: Vec<Entry<'a, B>>,
    /// Whether a validated model execution scope is currently active.
    running: bool,
    /// Registration count fixed by preflight to detect mid-run additions.
    run_entry_count: usize,
    /// Execution-scoped validity masks, applied even when no action is registered.
    validity: Vec<PointValidity<B>>,
}

impl<B: Backend> Default for Hooks<'_, B> {
    /// Creates an empty hook collection with its own capture-handle identity.
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, B: Backend> Hooks<'a, B> {
    /// Create an empty collection. No tensors are retained until requested.
    pub fn new() -> Self {
        Self {
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            entries: Vec::new(),
            running: false,
            run_entry_count: 0,
            validity: Vec::new(),
        }
    }

    /// Whether no captures or edits have been registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether a precise point is requested, allowing kernels to avoid unnecessary materialization.
    pub fn is_requested(&self, point: PointInfo) -> bool {
        self.entries.iter().any(|entry| entry.point == point)
    }

    /// Whether execution must visit a point for an action or scoped validity.
    ///
    /// Model and kernel dispatch should use this predicate when skipping an
    /// intermediate would also skip a declared validity invariant. Score/pattern
    /// points still require the implementation's explicit precision capability.
    pub fn needs_point(&self, point: PointInfo) -> bool {
        self.is_requested(point.clone()) || self.validity.iter().any(|mask| mask.point == point)
    }

    /// Whether this scope has registered actions or active validity invariants.
    /// Unlike [`Self::is_empty`], this includes execution-only validity masks.
    pub fn has_work(&self) -> bool {
        !self.entries.is_empty() || !self.validity.is_empty()
    }

    /// Capture the selected tensor on its device, detached from the gradient graph.
    /// Repeated visits retain the latest value; model-provided stateless points occur once.
    pub fn capture<const D: usize>(&mut self, selection: impl Into<Selection<D>>) -> Capture<D> {
        self.capture_impl(selection.into(), true)
    }

    /// Capture while retaining gradient connectivity and its associated memory.
    pub fn capture_with_grad<const D: usize>(
        &mut self,
        selection: impl Into<Selection<D>>,
    ) -> Capture<D> {
        self.capture_impl(selection.into(), false)
    }

    /// Registers a capture with an explicit graph-retention policy and a collection-bound handle.
    fn capture_impl<const D: usize>(
        &mut self,
        selection: Selection<D>,
        detach: bool,
    ) -> Capture<D> {
        let handle = Capture {
            owner: self.owner,
            entry: self.entries.len(),
        };
        self.entries.push(Entry::new(
            selection.point.info,
            selection.slices,
            Action::Capture {
                detach,
                saved: None,
            },
        ));
        handle
    }

    /// Capture a complete point of any rank, detached, for catalog-driven tools.
    ///
    /// Read the result with [`Self::get_dynamic`]. For example, capturing every
    /// point a model lists reproduces a full activation cache:
    ///
    /// ```
    /// use burn::{backend::Flex, tensor::Tensor};
    /// use transformers::intervention::{Hooks, Layer};
    ///
    /// let points = vec![Layer::decoder(0).mlp_output().info()];
    /// let mut hooks = Hooks::<Flex>::new();
    /// let handles = points
    ///     .iter()
    ///     .map(|point| hooks.capture_dynamic(point.clone()))
    ///     .collect::<Vec<_>>();
    /// hooks.run(points, |hooks| {
    ///     let device = Default::default();
    ///     hooks.apply(Layer::decoder(0).mlp_output(), Tensor::ones([1, 2, 4], &device))
    /// })?;
    /// let activation = hooks.get_dynamic(&handles[0])?;
    /// assert_eq!(activation.shape, [1, 2, 4]);
    /// # Ok::<(), transformers::Error>(())
    /// ```
    pub fn capture_dynamic(&mut self, point: PointInfo) -> DynamicCapture {
        let handle = DynamicCapture {
            owner: self.owner,
            entry: self.entries.len(),
            point: point.clone(),
        };
        self.entries.push(Entry::new(
            point,
            Vec::new(),
            Action::Capture {
                detach: true,
                saved: None,
            },
        ));
        handle
    }

    /// Transform a selected activation. The closure receives a rank-preserving slice.
    pub fn edit<const D: usize>(
        &mut self,
        selection: impl Into<Selection<D>>,
        mut edit: impl FnMut(Tensor<B, D>) -> Tensor<B, D> + 'a,
    ) {
        self.try_edit(selection, move |tensor| Ok(edit(tensor)));
    }

    /// Inspect a detached on-device selection without retaining it in this collection.
    ///
    /// Compute a reduction or record your own small result in the callback. The
    /// forward tensor and its gradient connectivity remain unchanged, and no
    /// slice assignment, validity re-application, or point constraint runs for
    /// an observer, so observing an attention pattern adds no quadratic work.
    pub fn observe<const D: usize>(
        &mut self,
        selection: impl Into<Selection<D>>,
        mut observe: impl FnMut(Tensor<B, D>) + 'a,
    ) {
        let selection = selection.into();
        let callback = move |tensor: Tensor<B, 1>, shape: Vec<usize>| {
            let shape: [usize; D] = shape.try_into().map_err(|_| {
                Error::configuration("hooks.rank", "observer rank does not match point")
            })?;
            observe(tensor.reshape(shape));
            Ok(())
        };
        self.entries.push(Entry::new(
            selection.point.info,
            selection.slices,
            Action::Observe(Box::new(callback)),
        ));
    }

    /// Transform a selected activation with an application-defined fallible callback.
    pub fn try_edit<const D: usize>(
        &mut self,
        selection: impl Into<Selection<D>>,
        mut edit: impl FnMut(Tensor<B, D>) -> Result<Tensor<B, D>> + 'a,
    ) {
        let selection = selection.into();
        let callback = move |tensor: Tensor<B, 1>, shape: Vec<usize>| {
            let shape: [usize; D] = shape.try_into().map_err(|_| {
                Error::configuration("hooks.rank", "callback rank does not match point")
            })?;
            let result = edit(tensor.reshape(shape))?;
            Ok(Saved {
                shape: result.dims().to_vec(),
                tensor: result.flatten(0, D - 1),
            })
        };
        self.entries.push(Entry::new(
            selection.point.info,
            selection.slices,
            Action::Edit(Box::new(callback)),
        ));
    }

    /// Replace a selected activation, for example with a saved clean-run tensor.
    pub fn replace<const D: usize>(
        &mut self,
        selection: impl Into<Selection<D>>,
        tensor: Tensor<B, D>,
    ) {
        self.edit(selection, move |_| tensor.clone());
    }

    /// Read a capture from the latest successful run. No implicit host transfer occurs.
    pub fn get<const D: usize>(&self, capture: &Capture<D>) -> Result<Tensor<B, D>> {
        let (point, saved) = self.saved(capture.owner, capture.entry)?;
        let shape: [usize; D] = saved
            .shape
            .clone()
            .try_into()
            .map_err(|_| Error::configuration(&point.name, "capture rank mismatch"))?;
        Ok(saved.tensor.clone().reshape(shape))
    }

    /// Read a run-time-rank capture from the latest successful run, still on device.
    pub fn get_dynamic(&self, capture: &DynamicCapture) -> Result<DynamicActivation<B>> {
        let (_, saved) = self.saved(capture.owner, capture.entry)?;
        Ok(DynamicActivation {
            values: saved.tensor.clone(),
            shape: saved.shape.clone(),
        })
    }

    /// Looks up a saved capture by owner and entry, rejecting foreign or unrun handles.
    fn saved(&self, owner: u64, entry: usize) -> Result<(&PointInfo, &Saved<B>)> {
        if owner != self.owner {
            return Err(Error::configuration(
                "hooks.capture",
                "capture belongs to another Hooks collection",
            ));
        }
        let entry = self
            .entries
            .get(entry)
            .ok_or_else(|| Error::configuration("hooks.capture", "unknown capture"))?;
        match &entry.action {
            Action::Capture {
                saved: Some(saved), ..
            } => Ok((&entry.point, saved)),
            _ => Err(Error::configuration(
                &entry.point.name,
                "capture is unavailable; complete a successful forward pass first",
            )),
        }
    }

    /// Execute a model's stateless scope, validating requested points before computation.
    /// Missing points, impossible axes, nested scopes, and unvisited points are errors.
    /// A failed run discards partial captures. Actions may be reused for another run.
    /// Register actions before starting the scope; adding actions during execution
    /// is rejected. Side effects in user callbacks are not rolled back on failure.
    pub fn run<T>(
        &mut self,
        points: Vec<PointInfo>,
        forward: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.run_with_validity(points, Vec::new(), forward)
    }

    /// Execute a stateless scope with point-specific activation validity.
    ///
    /// Validity is an execution invariant, independent of captures and edits.
    /// Invalid elements are zero before actions and after each edit, even when
    /// the point has no registered actions. Masks must name catalog points and
    /// are discarded after success or failure; an ordinary [`Self::run`] starts
    /// with no validity masks. Nested scopes remain unsupported.
    pub fn run_with_validity<T>(
        &mut self,
        points: Vec<PointInfo>,
        validity: Vec<PointValidity<B>>,
        forward: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        if self.running {
            return Err(Error::configuration(
                "hooks.scope",
                "nested hook scopes are unsupported",
            ));
        }
        self.clear_captures();
        self.validity.clear();
        for (index, mask) in validity.iter().enumerate() {
            if !points.contains(&mask.point) {
                return Err(Error::configuration(
                    &mask.point.name,
                    "activation validity point is unavailable on this execution path",
                ));
            }
            if validity[..index]
                .iter()
                .any(|previous| previous.point == mask.point)
            {
                return Err(Error::configuration(
                    &mask.point.name,
                    "each activation point may have only one validity mask",
                ));
            }
        }
        for entry in &self.entries {
            if !points.contains(&entry.point) {
                return Err(Error::configuration(
                    &entry.point.name,
                    "activation point is unavailable on this execution path; inspect activation_points()",
                ));
            }
            if entry.point.axes.is_empty() {
                return Err(Error::configuration(
                    &entry.point.name,
                    "activation rank must be positive",
                ));
            }
            entry.validate_selection()?;
        }
        self.validity = validity;
        self.running = true;
        self.run_entry_count = self.entries.len();
        let result = forward(self).and_then(|output| {
            self.check_registration_count()?;
            if let Some(entry) = self.entries.iter().find(|entry| entry.visits == 0) {
                Err(Error::configuration(
                    &entry.point.name,
                    "requested activation was not visited during this forward pass",
                ))
            } else {
                Ok(output)
            }
        });
        self.running = false;
        self.validity.clear();
        if result.is_err() {
            self.clear_captures();
        }
        result
    }

    /// Rejects registrations added during execution because the validated scope must remain fixed.
    fn check_registration_count(&self) -> Result<()> {
        if self.entries.len() != self.run_entry_count {
            Err(Error::configuration(
                "hooks.scope",
                "register captures and edits before starting the forward pass",
            ))
        } else {
            Ok(())
        }
    }

    /// Releases captured tensors and resets visit counts before a run or after failure.
    fn clear_captures(&mut self) {
        for entry in &mut self.entries {
            entry.visits = 0;
            if let Action::Capture { saved, .. } = &mut entry.action {
                *saved = None;
            }
        }
    }

    /// Emit a typed point from a model implementation. Public model wrappers call
    /// this inside [`Hooks::run`]. Unrequested tensors pass through unchanged.
    pub fn apply<const D: usize>(
        &mut self,
        point: Point<D>,
        tensor: Tensor<B, D>,
    ) -> Result<Tensor<B, D>> {
        self.apply_constrained(point, tensor, Ok)
    }

    /// Emit a point while restoring a model invariant after each edit, before
    /// subsequent actions see the value. For example, attention implementations
    /// reapply excluded links after a pattern edit. The constraint receives the
    /// full tensor after any selected slice has been assigned back; it must
    /// preserve shape, dtype, and device. The incoming tensor must already obey
    /// the invariant. Captures alone do not invoke the constraint.
    pub fn apply_constrained<const D: usize>(
        &mut self,
        point: Point<D>,
        mut tensor: Tensor<B, D>,
        mut constraint: impl FnMut(Tensor<B, D>) -> Result<Tensor<B, D>>,
    ) -> Result<Tensor<B, D>> {
        if self.entries.is_empty() && self.validity.is_empty() {
            return Ok(tensor);
        }
        if !self.running {
            return Err(Error::configuration(
                "hooks.scope",
                "emit activations inside Hooks::run",
            ));
        }
        self.check_registration_count()?;
        let validity = self.validity.iter().find(|mask| mask.point == point.info);
        if let Some(validity) = validity {
            tensor = validity.apply(tensor)?;
        }
        for entry in &mut self.entries {
            if entry.point != point.info {
                continue;
            }
            tensor = apply_entry(entry, tensor, validity, &mut constraint)?;
            entry.visits += 1;
        }
        Ok(tensor)
    }
}

impl<B: Backend> Hooks<'_, B> {
    /// Emits the derived sink-probability point of an attention pattern that has just been emitted.
    ///
    /// `pattern` is the effective `[batch, head, query, key]` attention
    /// pattern after any edits (the value [`Hooks::apply_constrained`]
    /// returned for [`AttentionPoints::pattern`]), and `points` addresses the
    /// same attention operation. When the sink point is requested, this
    /// computes `p_sink = 1 - sum_k pattern` in F32 (the subtraction cancels
    /// almost all of the sum, which half precision would lose) and emits it as
    /// `[batch, head, query]`; otherwise it does no work. Captures and
    /// observers see the value; edits fail with [`Error::Intervention`],
    /// because the sink mass is derived and changing it would not change the
    /// attention output. Call it only for attention with learned sinks, inside
    /// the same hook scope as the pattern.
    ///
    /// ```
    /// use burn::{backend::Flex, tensor::Tensor};
    /// use transformers::intervention::{Hooks, Layer};
    ///
    /// let points = Layer::decoder(0).self_attention();
    /// let mut hooks = Hooks::<Flex>::new();
    /// let sink = hooks.capture(points.sink_probability());
    /// let pattern = Tensor::<Flex, 4>::from_floats([[[[0.5, 0.25], [0.0, 0.0]]]], &Default::default());
    /// hooks.run(vec![points.pattern().info(), points.sink_probability().info()], |hooks| {
    ///     let pattern = hooks.apply(points.pattern(), pattern)?;
    ///     hooks.apply_sink_probability(points, &pattern)
    /// })?;
    /// assert_eq!(hooks.get(&sink)?.into_data().to_vec::<f32>().unwrap(), [0.25, 1.0]);
    /// # Ok::<(), transformers::Error>(())
    /// ```
    pub fn apply_sink_probability(
        &mut self,
        points: AttentionPoints,
        pattern: &Tensor<B, 4>,
    ) -> Result<()> {
        let point = points.sink_probability();
        if !self.needs_point(point.info()) {
            return Ok(());
        }
        let [batch, heads, queries, _] = pattern.dims();
        let mass = pattern
            .clone()
            .cast(burn::tensor::DType::F32)
            .sum_dim(3)
            .reshape([batch, heads, queries]);
        let sink = mass.neg().add_scalar(1.0);
        let name = point.info().name;
        self.apply_constrained(point, sink, |_| {
            Err(Error::intervention(
                &name,
                "the sink probability is derived from the attention pattern and read-only; edit the pattern or the sink logits instead",
            ))
        })
        .map(|_| ())
    }
}

/// Applies one registered action to an emitted activation and returns the result.
///
/// Captures and observers read the selection and return the activation
/// unchanged. Edits replace the selection, then re-apply validity and the
/// point's constraint; every failure names the point as an intervention error.
fn apply_entry<B: Backend, const D: usize>(
    entry: &mut Entry<'_, B>,
    tensor: Tensor<B, D>,
    validity: Option<&PointValidity<B>>,
    constraint: &mut impl FnMut(Tensor<B, D>) -> Result<Tensor<B, D>>,
) -> Result<Tensor<B, D>> {
    let shape = tensor.dims();
    let ranges = selection_ranges(&shape, &entry.resolved).map_err(|index| {
        let (axis, range) = &entry.slices[index];
        let dimension = entry.resolved[index].dimension;
        // Preflight resolves every axis to a declared dimension, so the
        // missing-dimension arm only guards a point emitted at the wrong rank.
        let reason = match shape.get(dimension) {
            Some(extent) => format!(
                "{axis:?} selection {range:?} is outside extent {extent} of the emitted {shape:?} activation"
            ),
            None => format!(
                "{axis:?} selection resolves to dimension {dimension}, which the emitted {shape:?} activation lacks"
            ),
        };
        Error::intervention(&entry.point.name, reason)
    })?;
    let ranges: [Range<usize>; D] = ranges.try_into().map_err(|_| {
        Error::intervention(
            &entry.point.name,
            "selection rank differs from the activation",
        )
    })?;
    let selected = if entry.slices.is_empty() {
        tensor.clone()
    } else {
        tensor.clone().slice(ranges.clone())
    };
    let selected_shape = selected.dims();
    match &mut entry.action {
        Action::Capture { detach, saved } => {
            let selected = if *detach { selected.detach() } else { selected };
            *saved = Some(Saved {
                tensor: selected.flatten(0, D - 1),
                shape: selected_shape.to_vec(),
            });
            Ok(tensor)
        }
        Action::Observe(observe) => {
            observe(selected.detach().flatten(0, D - 1), selected_shape.to_vec()).map_err(
                |error| Error::intervention_caused_by(&entry.point.name, "observer failed", error),
            )?;
            Ok(tensor)
        }
        Action::Edit(edit) => {
            let dtype = selected.dtype();
            let device = selected.device();
            let changed =
                edit(selected.flatten(0, D - 1), selected_shape.to_vec()).map_err(|error| {
                    Error::intervention_caused_by(
                        &entry.point.name,
                        "edit callback returned an error",
                        error,
                    )
                })?;
            if changed.shape != selected_shape
                || changed.tensor.dtype() != dtype
                || changed.tensor.device() != device
            {
                return Err(Error::intervention(
                    &entry.point.name,
                    format!(
                        "replacement must preserve selected shape {selected_shape:?}, dtype {dtype:?}, and device; found shape {:?}, dtype {:?}. Use an explicit module boundary for dimension changes",
                        changed.shape,
                        changed.tensor.dtype()
                    ),
                ));
            }
            let changed = changed.tensor.reshape(selected_shape);
            let mut tensor = if entry.slices.is_empty() {
                changed
            } else {
                tensor.slice_assign(ranges, changed)
            };
            if let Some(validity) = validity {
                tensor = validity.apply(tensor)?;
            }
            tensor = constraint(tensor).map_err(|error| {
                Error::intervention_caused_by(
                    &entry.point.name,
                    "activation constraint failed",
                    error,
                )
            })?;
            if tensor.dims() != shape || tensor.dtype() != dtype || tensor.device() != device {
                return Err(Error::intervention(
                    &entry.point.name,
                    "activation constraint must preserve full shape, dtype, and device",
                ));
            }
            match validity {
                Some(validity) => validity.apply(tensor),
                None => Ok(tensor),
            }
        }
    }
}

#[cfg(test)]
#[path = "intervention_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "intervention/validity_tests.rs"]
mod validity_tests;
