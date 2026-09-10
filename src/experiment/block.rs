//! Open residual blocks and explicit width changes built from ordinary Burn modules.

use burn::{
    module::{Module, ModuleDisplay},
    nn::{Linear, LinearConfig, LinearLayout},
    tensor::{Tensor, backend::Backend},
};

use crate::{
    Error, Result,
    architecture::{NormPlacement, NormalizationSpec},
    intervention::{Hooks, Layer},
    models::custom::NormLayout,
    operations,
};

use super::{ExecutionContext, HiddenTransform};

/// A transformer block whose mixer and feed-forward computation are ordinary Rust modules.
///
/// Implement [`HiddenTransform`] for your own attention, sequence mixer, or small model,
/// then place it in either branch. Both modules and every normalization parameter belong
/// to this module tree and therefore participate in recording, freezing, and optimization.
/// Each branch must return the same `[batch, token, channel]` shape it receives.
#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend, M: Module<B>, F: Module<B>> {
    /// Replaceable sequence mixer, such as attention or a tokenwise small model.
    pub mixer: M,
    /// Replaceable feed-forward computation.
    pub feed_forward: F,
    /// Registered normalization around the mixer branch.
    pub mixer_norm: NormLayout<B>,
    /// Registered normalization around the feed-forward branch.
    pub feed_forward_norm: NormLayout<B>,
    /// Residual channel width accepted and returned by this block.
    #[module(skip)]
    pub hidden_size: usize,
}

impl<B: Backend, M: HiddenTransform<B> + ModuleDisplay, F: HiddenTransform<B> + ModuleDisplay>
    ResidualBlock<B, M, F>
{
    /// Executes this registered composition with explicit context and no observers.
    pub fn run(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
    ) -> Result<Tensor<B, 3>> {
        self.run_with_hooks(hidden, context, layer, &mut Hooks::new())
    }

    /// Executes this composition in a scope containing its declared inner points.
    pub fn run_with_hooks(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        hooks.run(self.activation_points(layer), |hooks| {
            self.forward(hidden, context, layer, hooks)
        })
    }

    /// Builds `x + mixer(x)` followed by `x + feed_forward(x)`, with no normalization.
    ///
    /// This makes the unnormalized architecture used by small circuit experiments an
    /// explicit choice; no normalization parameters are allocated or hidden in callbacks.
    pub fn without_normalization(hidden_size: usize, mixer: M, feed_forward: F) -> Result<Self> {
        if hidden_size == 0 {
            return Err(Error::configuration(
                "residual_block.hidden_size",
                "must be non-zero",
            ));
        }
        Ok(Self {
            mixer,
            feed_forward,
            mixer_norm: NormLayout::identity(),
            feed_forward_norm: NormLayout::identity(),
            hidden_size,
        })
    }

    /// Builds the residual wiring around explicitly supplied modules.
    ///
    /// `Pre` evaluates `x + mixer(norm(x))`, then the analogous feed-forward branch;
    /// `Post` and `Sandwich` follow [`NormLayout`]. Modules own their own input/output
    /// projections, and a shape mismatch returns an error before residual addition.
    pub fn new(
        hidden_size: usize,
        mixer: M,
        feed_forward: F,
        normalization: NormalizationSpec,
        placement: NormPlacement,
        device: &B::Device,
    ) -> Result<Self> {
        if hidden_size == 0 {
            return Err(Error::configuration(
                "residual_block.hidden_size",
                "must be non-zero",
            ));
        }
        normalization.validate("residual_block.normalization")?;
        Ok(Self {
            mixer,
            feed_forward,
            mixer_norm: NormLayout::init(normalization, placement, hidden_size, device)?,
            feed_forward_norm: NormLayout::init(normalization, placement, hidden_size, device)?,
            hidden_size,
        })
    }
}

impl<B: Backend, M: HiddenTransform<B> + ModuleDisplay, F: HiddenTransform<B> + ModuleDisplay>
    HiddenTransform<B> for ResidualBlock<B, M, F>
{
    fn activation_points(&self, layer: Layer) -> Vec<crate::intervention::PointInfo> {
        let mut points = vec![layer.residual_pre().info(), layer.attention_input().info()];
        points.extend(self.mixer.activation_points(layer));
        points.extend([layer.attention_output().info(), layer.mlp_input().info()]);
        points.extend(self.feed_forward.activation_points(layer));
        points.extend([layer.mlp_output().info(), layer.residual_post().info()]);
        points
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        validate_width(&hidden, self.hidden_size, "residual_block.input")?;
        context.validate([hidden.dims()[0], hidden.dims()[1]], &hidden.device())?;
        let hidden = hooks.apply(layer.residual_pre(), hidden)?;
        let residual = hidden.clone();
        let prepared = hooks.apply(layer.attention_input(), self.mixer_norm.prepare(hidden)?)?;
        let mixed = self.mixer.forward(prepared, context, layer, hooks)?;
        validate_same_shape(&mixed, &residual, "residual_block.mixer")?;
        let mixed = hooks.apply(layer.attention_output(), mixed)?;
        let hidden = self.mixer_norm.finish(residual, mixed)?;
        let residual = hidden.clone();
        let prepared = hooks.apply(layer.mlp_input(), self.feed_forward_norm.prepare(hidden)?)?;
        let projected = self.feed_forward.forward(prepared, context, layer, hooks)?;
        validate_same_shape(&projected, &residual, "residual_block.feed_forward")?;
        let projected = hooks.apply(layer.mlp_output(), projected)?;
        let hidden = self.feed_forward_norm.finish(residual, projected)?;
        hooks.apply(layer.residual_post(), hidden)
    }
}

/// Wrap a module with explicit, registered projections between residual widths.
///
/// `input_size -> inner_size -> output_size` makes every width change reviewable.
/// Equal adjacent widths omit the corresponding projection. To assemble a heterogeneous
/// stack, use one wrapper per stage and feed each output into the next stage normally.
#[derive(Module, Debug)]
pub struct WidthAdapter<B: Backend, T: Module<B>> {
    /// Projection into the wrapped module's channel width, absent when widths match.
    pub input_projection: Option<Linear<B>>,
    /// Ordinary module that computes at `inner_size` channels.
    pub transform: T,
    /// Projection into the next stage's channel width, absent when widths match.
    pub output_projection: Option<Linear<B>>,
    /// Accepted residual width.
    #[module(skip)]
    pub input_size: usize,
    /// Width consumed and returned by the wrapped module.
    #[module(skip)]
    pub inner_size: usize,
    /// Returned residual width.
    #[module(skip)]
    pub output_size: usize,
}

impl<B: Backend, T: HiddenTransform<B> + ModuleDisplay> WidthAdapter<B, T> {
    /// Executes this registered composition with explicit context and no observers.
    pub fn run(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
    ) -> Result<Tensor<B, 3>> {
        self.run_with_hooks(hidden, context, layer, &mut Hooks::new())
    }

    /// Executes this composition in a scope containing its declared inner points.
    pub fn run_with_hooks(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        hooks.run(self.activation_points(layer), |hooks| {
            self.forward(hidden, context, layer, hooks)
        })
    }

    /// Allocates only the explicitly requested channel changes, using bias-free projections.
    pub fn new(
        transform: T,
        input_size: usize,
        inner_size: usize,
        output_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        if input_size == 0 || inner_size == 0 || output_size == 0 {
            return Err(Error::configuration(
                "width_adapter.dimensions",
                "input, inner, and output widths must be non-zero",
            ));
        }
        let projection = |input, output| {
            operations::init_linear(
                &LinearConfig::new(input, output)
                    .with_bias(false)
                    .with_layout(LinearLayout::Col),
                device,
            )
        };
        Ok(Self {
            input_projection: (input_size != inner_size)
                .then(|| projection(input_size, inner_size)),
            transform,
            output_projection: (inner_size != output_size)
                .then(|| projection(inner_size, output_size)),
            input_size,
            inner_size,
            output_size,
        })
    }
}

impl<B: Backend, T: HiddenTransform<B> + ModuleDisplay> HiddenTransform<B> for WidthAdapter<B, T> {
    fn activation_points(&self, layer: Layer) -> Vec<crate::intervention::PointInfo> {
        self.transform.activation_points(layer)
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        validate_width(&hidden, self.input_size, "width_adapter.input")?;
        let [batch, tokens, _] = hidden.dims();
        let device = hidden.device();
        let dtype = hidden.dtype();
        context.validate([batch, tokens], &device)?;
        let hidden = match &self.input_projection {
            Some(projection) => operations::linear(hidden, projection)?,
            None => hidden,
        };
        let hidden = self.transform.forward(hidden, context, layer, hooks)?;
        if hidden.dims() != [batch, tokens, self.inner_size] {
            return Err(Error::shape(
                "width_adapter.transform",
                format!("[{batch}, {tokens}, {}]", self.inner_size),
                format!("{:?}", hidden.dims()),
            ));
        }
        if hidden.device() != device || hidden.dtype() != dtype {
            return Err(Error::configuration(
                "width_adapter.transform",
                "the wrapped module must preserve device and dtype; width adapters only change channel dimensions",
            ));
        }
        match &self.output_projection {
            Some(projection) => operations::linear(hidden, projection),
            None => Ok(hidden),
        }
    }
}

fn validate_width<B: Backend>(hidden: &Tensor<B, 3>, width: usize, field: &str) -> Result<()> {
    let [batch, tokens, actual] = hidden.dims();
    if batch == 0 || tokens == 0 || actual != width {
        return Err(Error::shape(
            field,
            format!("[batch>0, token>0, {width}]"),
            format!("{:?}", hidden.dims()),
        ));
    }
    Ok(())
}

fn validate_same_shape<B: Backend>(
    actual: &Tensor<B, 3>,
    expected: &Tensor<B, 3>,
    field: &str,
) -> Result<()> {
    if actual.dims() != expected.dims()
        || actual.device() != expected.device()
        || actual.dtype() != expected.dtype()
    {
        return Err(Error::configuration(
            field,
            "a residual branch must preserve batch, token, and channel dimensions, device, and dtype",
        ));
    }
    Ok(())
}

impl<B: Backend> HiddenTransform<B> for Linear<B> {
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        _context: &ExecutionContext<B>,
        _layer: Layer,
        _hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        operations::linear(hidden, self)
    }
}

impl<B: Backend> HiddenTransform<B> for crate::feed_forward::DenseFeedForward<B> {
    fn activation_points(&self, layer: Layer) -> Vec<crate::intervention::PointInfo> {
        vec![
            layer.mlp_pre_activation().info(),
            layer.mlp_activation().info(),
        ]
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        _context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let hidden = hooks.apply(
            layer.mlp_pre_activation(),
            operations::linear(hidden, &self.up_proj)?,
        )?;
        let hidden = crate::feed_forward::apply_activation(hidden, self.activation);
        let hidden = hooks.apply(layer.mlp_activation(), hidden)?;
        operations::linear(hidden, &self.down_proj)
    }
}

impl<B: Backend> HiddenTransform<B> for crate::feed_forward::GatedFeedForward<B> {
    fn activation_points(&self, layer: Layer) -> Vec<crate::intervention::PointInfo> {
        vec![
            layer.mlp_pre_activation().info(),
            layer.mlp_activation().info(),
        ]
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        _context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let gate = hooks.apply(
            layer.mlp_pre_activation(),
            operations::linear(hidden.clone(), &self.gate_proj)?,
        )?;
        let gate = crate::feed_forward::apply_activation(gate, self.activation);
        let gate = hooks.apply(layer.mlp_activation(), gate)?;
        let content = operations::linear(hidden, &self.up_proj)?;
        operations::linear(gate * content, &self.down_proj)
    }
}

impl<B: Backend> HiddenTransform<B> for crate::attention::GroupedQueryAttention<B> {
    fn activation_points(&self, layer: Layer) -> Vec<crate::intervention::PointInfo> {
        let points = layer.self_attention();
        if self.precision_mode == crate::attention::AttentionPrecisionMode::EagerFp32Softmax {
            points.all()
        } else {
            vec![
                points.query().info(),
                points.key().info(),
                points.value().info(),
                points.head_context().info(),
            ]
        }
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        context: &ExecutionContext<B>,
        layer: Layer,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, tokens, _] = hidden.dims();
        context.validate([batch, tokens], &hidden.device())?;
        let policy =
            context.attention_for(layer, layer.stack() == crate::intervention::Stack::Decoder);
        let shape = [batch, self.query_heads, tokens, tokens];
        policy.validate(shape, &hidden.device())?;
        if policy
            .bias
            .as_ref()
            .is_some_and(|bias| bias.dtype() != hidden.dtype())
        {
            return Err(Error::configuration(
                "residual_block.attention.bias",
                "bias and input must use the same dtype",
            ));
        }
        let padding = crate::attention::padding_attention_mask(
            context.token_mask.as_ref(),
            context.token_mask.as_ref(),
            batch,
            self.query_heads,
            tokens,
            tokens,
        )?;
        let mask = policy
            .exclusion_mask(padding, shape, &hidden.device())?
            .map(|mask| mask.expand(shape));
        let bias = policy.expanded_bias(shape, &hidden.device())?;
        self.attend_with_hooks(
            self.project_self(hidden)?,
            mask,
            bias,
            policy.causal,
            layer.self_attention(),
            hooks,
        )
    }
}
