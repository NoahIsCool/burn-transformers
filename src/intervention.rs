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

mod validity;
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

/// Discoverable description of a computation point, independent of tensor rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointInfo {
    /// Stable semantic name, such as `decoder.layers.0.mlp.activation`.
    pub name: String,
    /// Dimension meanings in their actual tensor order.
    pub axes: Vec<Axis>,
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
    /// A saturated endpoint becomes an invalid empty range and is rejected during
    /// scope validation rather than panicking during registration.
    fn one(self, axis: Axis, index: usize) -> Self {
        self.slice(axis, index..index.saturating_add(1))
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
    /// Per-query-head context before output gating and output projection.
    pub fn head_context(self) -> Point<4> {
        self.point(
            "head_context",
            [Axis::Batch, Axis::Head, Axis::Token, Axis::Channel],
        )
    }
    /// Catalog the six attention points, without claiming a model supports them.
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

/// Rank-independent captured activation retained with its original shape.
struct Saved<B: Backend> {
    /// Flattened activation, detached only when requested by the capture policy.
    tensor: Tensor<B, 1>,
    /// Runtime dimensions restored when reading or editing the capture.
    shape: Vec<usize>,
}
/// Borrowed mutable callback receiving a flattened activation and its runtime shape.
type Edit<'a, B> = Box<dyn FnMut(Tensor<B, 1>, Vec<usize>) -> Result<Saved<B>> + 'a>;
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
}
/// Registered point selection and action together with run-local visit accounting.
struct Entry<'a, B: Backend> {
    /// Point name and semantic axes used during preflight and execution.
    point: PointInfo,
    /// Optional rank-preserving ranges to capture or replace.
    slices: Vec<(Axis, Range<usize>)>,
    /// Observer or edit to perform when the selected point is reached.
    action: Action<'a, B>,
    /// Number of emissions seen during the current scoped execution.
    visits: usize,
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
        self.entries.push(Entry {
            point: selection.point.info,
            slices: selection.slices,
            action: Action::Capture {
                detach,
                saved: None,
            },
            visits: 0,
        });
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
    /// Compute a reduction or record your own small result in the callback. The
    /// original forward tensor and its gradient connectivity remain unchanged.
    pub fn observe<const D: usize>(
        &mut self,
        selection: impl Into<Selection<D>>,
        mut observe: impl FnMut(Tensor<B, D>) + 'a,
    ) {
        self.edit(selection, move |tensor| {
            observe(tensor.clone().detach());
            tensor
        });
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
        self.entries.push(Entry {
            point: selection.point.info,
            slices: selection.slices,
            action: Action::Edit(Box::new(callback)),
            visits: 0,
        });
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
        if capture.owner != self.owner {
            return Err(Error::configuration(
                "hooks.capture",
                "capture belongs to another Hooks collection",
            ));
        }
        let entry = self
            .entries
            .get(capture.entry)
            .ok_or_else(|| Error::configuration("hooks.capture", "unknown capture"))?;
        match &entry.action {
            Action::Capture {
                saved: Some(saved), ..
            } => {
                let shape: [usize; D] = saved.shape.clone().try_into().map_err(|_| {
                    Error::configuration(&entry.point.name, "capture rank mismatch")
                })?;
                Ok(saved.tensor.clone().reshape(shape))
            }
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
            validate_slices(&entry.point, &entry.slices)?;
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
            let shape = tensor.dims();
            let mut ranges: [Range<usize>; D] = std::array::from_fn(|axis| 0..shape[axis]);
            for (axis, range) in &entry.slices {
                let index = entry
                    .point
                    .axes
                    .iter()
                    .position(|candidate| candidate == axis)
                    .ok_or_else(|| {
                        Error::configuration(
                            &entry.point.name,
                            format!("point has no {axis:?} axis"),
                        )
                    })?;
                if range.start >= range.end || range.end > shape[index] {
                    return Err(Error::configuration(
                        &entry.point.name,
                        format!(
                            "{axis:?} selection {range:?} is outside extent {}",
                            shape[index]
                        ),
                    ));
                }
                ranges[index] = range.clone();
            }
            let selected = if entry.slices.is_empty() {
                tensor.clone()
            } else {
                tensor.clone().slice(ranges.clone())
            };
            let selected_shape = selected.dims();
            let dtype = selected.dtype();
            let device = selected.device();
            match &mut entry.action {
                Action::Capture { detach, saved } => {
                    let selected = if *detach { selected.detach() } else { selected };
                    *saved = Some(Saved {
                        tensor: selected.flatten(0, D - 1),
                        shape: selected_shape.to_vec(),
                    });
                }
                Action::Edit(edit) => {
                    let changed = edit(selected.flatten(0, D - 1), selected_shape.to_vec())
                        .map_err(|error| {
                            Error::configuration(
                                &entry.point.name,
                                format!("intervention failed: {error}"),
                            )
                        })?;
                    if changed.shape != selected_shape
                        || changed.tensor.dtype() != dtype
                        || changed.tensor.device() != device
                    {
                        return Err(Error::configuration(
                            &entry.point.name,
                            format!(
                                "replacement must preserve selected shape {selected_shape:?}, dtype {dtype:?}, and device; use an explicit module boundary for dimension changes"
                            ),
                        ));
                    }
                    let changed = changed.tensor.reshape(selected_shape);
                    tensor = if entry.slices.is_empty() {
                        changed
                    } else {
                        tensor.slice_assign(ranges, changed)
                    };
                    if let Some(validity) = validity {
                        tensor = validity.apply(tensor)?;
                    }
                    tensor = constraint(tensor).map_err(|error| {
                        Error::configuration(
                            &entry.point.name,
                            format!("activation constraint failed: {error}"),
                        )
                    })?;
                    if tensor.dims() != shape
                        || tensor.dtype() != dtype
                        || tensor.device() != device
                    {
                        return Err(Error::configuration(
                            &entry.point.name,
                            "activation constraint must preserve full shape, dtype, and device",
                        ));
                    }
                    if let Some(validity) = validity {
                        tensor = validity.apply(tensor)?;
                    }
                }
            }
            entry.visits += 1;
        }
        Ok(tensor)
    }
}

/// Checks that selected axes exist exactly once and have unique, nonempty ranges.
fn validate_slices(point: &PointInfo, slices: &[(Axis, Range<usize>)]) -> Result<()> {
    for (index, (axis, range)) in slices.iter().enumerate() {
        if point
            .axes
            .iter()
            .filter(|candidate| *candidate == axis)
            .count()
            != 1
        {
            return Err(Error::configuration(
                &point.name,
                format!(
                    "selection needs exactly one {axis:?} axis; found {:?}",
                    point.axes
                ),
            ));
        }
        if slices[..index].iter().any(|(previous, _)| previous == axis) || range.start >= range.end
        {
            return Err(Error::configuration(
                &point.name,
                format!("duplicate or empty {axis:?} selection"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "intervention_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "intervention/validity_tests.rs"]
mod validity_tests;
