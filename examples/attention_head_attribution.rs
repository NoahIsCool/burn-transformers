//! Reproduce the additive head decomposition in Elhage et al. (2021).
//!
//! Run: cargo run --release --locked --example attention_head_attribution
//!
//! This is a mathematical reproduction on fixed, readable weights, NOT evidence
//! that a trained language model learned a particular circuit. One attention-only
//! layer, no normalization/MLP, and a linear readout make attribution exact:
//! logits = input @ unembedding + sum(head_output @ unembedding).
//! See examples/README.md for the paper, assumptions, and experimental scope.
//!
//! The model is a width-four F32 attention layer with two width-two heads and
//! a linear readout. Scores use `[batch,head,query,key]` axes and
//! `Q K^T / sqrt(head_width)`. Causal exclusion masks only future keys, so each
//! row keeps its own token; the bidirectional variant keeps every key.
//! Projected head outputs occupy residual channels `[batch,token,4]`, making
//! their logit contributions additive until readout. No nonlinear suffix,
//! padding, or cache is present.

use burn::{
    backend::Flex,
    module::Param,
    tensor::{Bool, Tensor, TensorData, activation::softmax},
};
use transformers::{
    attention::{GroupedQueryAttention, GroupedQueryAttentionConfig},
    intervention::{Axis, Hooks, Layer, Point, PointInfo},
};

/// CPU tensor backend for deterministic F32 attribution calculations.
type B = Flex<f32, i32>;
/// Residual states with axes `[batch, token, channel]`.
type Hidden = Tensor<B, 3>; // [batch, token, residual channel]
/// Maximum absolute discrepancy allowed in the attribution identities.
const TOLERANCE: f32 = 2e-5;

/// Expose one head after its output projection, with residual-channel axes.
/// The tensor is `[batch,token,4]`, unlike unprojected shared head context.
fn head_output(head: usize) -> Point<3> {
    Point::custom(
        format!("attribution.heads.{head}.projected_output"),
        [Axis::Batch, Axis::Token, Axis::Channel],
    )
}

/// Attention visibility used when comparing head contributions.
#[derive(Clone, Copy)]
enum Visibility {
    /// Allow each token to attend only to itself and earlier positions.
    Causal,
    /// Allow each token to attend to every position in the unpadded sequence.
    Bidirectional,
}

/// Fixed-weight attention model with a linear vocabulary readout.
struct TinyTransformer {
    /// Two-head attention whose projected contributions are captured separately.
    attention: GroupedQueryAttention<B>,
    /// Readout weights with axes `[residual channel, vocabulary]`.
    unembedding: Tensor<B, 2>,
}

impl TinyTransformer {
    /// Construct fixed Q/K/V, output-projection, and readout matrices in F32.
    /// All initializer values are replaced, so results have no RNG dependence.
    fn new() -> transformers::Result<Self> {
        let device = Default::default();
        let mut attention = GroupedQueryAttentionConfig::new(4, 2, 2, 2).init(&device)?;
        // Runtime linear weights use [input, output]. Every random initializer
        // is replaced: this example has no seed, download, or training dependency.
        let identity = Tensor::<B, 2>::from_floats(
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
            &device,
        );
        attention.q_proj.weight = Param::from_tensor(identity.clone());
        attention.k_proj.weight = Param::from_tensor(identity.clone());
        attention.v_proj.weight = Param::from_tensor(identity);
        attention.o_proj.weight = Param::from_tensor(Tensor::from_floats(
            [
                [1., 0.2, 0., 0.],
                [0.1, 1., 0., 0.],
                [0., 0., 1., 0.3],
                [0., 0., 0.2, 1.],
            ],
            &device,
        ));
        let unembedding = Tensor::from_floats(
            [[1., -1., 0.], [0., 1., -1.], [1., 0., -1.], [-1., 1., 0.5]],
            &device,
        );
        Ok(Self {
            attention,
            unembedding,
        })
    }

    /// Map `[batch,token,4]` residuals to `[batch,token,3]` logits linearly.
    /// Linearity is what makes summed head attributions exact in this model.
    fn readout(&self, hidden: Hidden) -> Hidden {
        hidden.matmul(self.unembedding.clone().unsqueeze())
    }

    /// Declare each projected head output and the post-attention residual point.
    fn activation_points(&self) -> Vec<PointInfo> {
        let mut points: Vec<_> = (0..self.attention.query_heads)
            .map(|head| head_output(head).info())
            .collect();
        points.push(Layer::decoder(0).residual_post().info());
        points
    }

    /// Project Q/K/V, apply the selected visibility, and expose each OV contribution.
    /// Softmax normalizes the key axis; causal masks exclude `key > query`.
    /// Head-output edits occur before summation into the residual, so their
    /// effects pass through only the fixed linear readout.
    fn forward_with_hooks(
        &self,
        input: Hidden,
        visibility: Visibility,
        hooks: &mut Hooks<'_, B>,
    ) -> transformers::Result<Hidden> {
        hooks.run(self.activation_points(), |hooks| {
            let projection = self.attention.project_self(input.clone())?;
            let [batch, heads, sequence, width] = projection.query.dims();
            let mut scores =
                projection.query.matmul(projection.key.transpose()) / (width as f64).sqrt();
            if matches!(visibility, Visibility::Causal) {
                let excluded = (0..sequence)
                    .flat_map(|query| (0..sequence).map(move |key| key > query))
                    .collect::<Vec<_>>();
                let mask = Tensor::<B, 4, Bool>::from_data(
                    TensorData::new(excluded, [1, 1, sequence, sequence]),
                    &input.device(),
                );
                scores = scores.mask_fill(mask, f32::NEG_INFINITY);
            }
            let patterns = softmax(scores, 3);
            let contexts = patterns.matmul(projection.value);
            let mut residual = input;
            for head in 0..heads {
                let context = contexts
                    .clone()
                    .narrow(1, head, 1)
                    .reshape([batch, sequence, width]);
                let output_weight = self
                    .attention
                    .o_proj
                    .weight
                    .val()
                    .narrow(0, head * width, width)
                    .unsqueeze();
                let contribution = context.matmul(output_weight);
                residual = residual + hooks.apply(head_output(head), contribution)?;
            }
            let residual = hooks.apply(Layer::decoder(0).residual_post(), residual)?;
            Ok(self.readout(residual))
        })
    }
}

/// Return the maximum absolute F32 difference, rejecting nonfinite comparisons.
fn max_error(left: Hidden, right: Hidden) -> f32 {
    let errors = (left - right).abs().to_data().to_vec::<f32>().unwrap();
    assert!(errors.iter().all(|error| error.is_finite()));
    errors.into_iter().fold(0., f32::max)
}

/// Check exact additive attribution, head ablation, restoration, and wrong-head control.
/// Also compare ordinary attention parity and the first token’s sensitivity
/// to a future-token edit under causal versus bidirectional visibility.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TinyTransformer::new()?;
    let input = Tensor::from_floats(
        [[
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [1., 0., 1., 0.],
            [0., 0., 0., 1.],
        ]],
        &Default::default(),
    );

    // 1. Observe selected intermediate tensors during an unchanged forward.
    // Captures stay on device and detach from autograd by default.
    let mut observe = Hooks::new();
    let head_zero = observe.capture(head_output(0));
    let head_one = observe.capture(head_output(1));
    let clean = model.forward_with_hooks(input.clone(), Visibility::Causal, &mut observe)?;
    let head_zero_output = observe.get(&head_zero)?;
    let head_one_output = observe.get(&head_one)?;

    // 2. Independently compare with the crate's ordinary attention execution.
    let production_attention = model.attention.attend(
        model.attention.project_self(input.clone())?,
        None,
        None,
        true,
    )?;
    let production = model.readout(input.clone() + production_attention);
    let mut parity_error = max_error(clean.clone(), production);
    assert!(
        parity_error < TOLERANCE,
        "instrumented/ordinary forward parity"
    );

    // 3. Reproduce the paper's exact additive decomposition into OV paths.
    let head_zero_logits = model.readout(head_zero_output.clone());
    let head_one_logits = model.readout(head_one_output.clone());
    let decomposition = model.readout(input.clone()) + head_zero_logits.clone() + head_one_logits;
    let decomposition_error = max_error(clean.clone(), decomposition);
    assert!(
        decomposition_error < TOLERANCE,
        "additive logit decomposition"
    );

    // 4. Intervene on head zero. Its direct attribution predicts the exact
    // logit change here because there is no later nonlinear computation.
    let mut ablate = Hooks::new();
    ablate.edit(head_output(0), |tensor| tensor.zeros_like());
    let ablated = model.forward_with_hooks(input.clone(), Visibility::Causal, &mut ablate)?;
    let ablation_effect = max_error(clean.clone(), ablated.clone());
    assert!(
        ablation_effect > 0.1,
        "the intervention must have a real effect"
    );
    let attribution_error = max_error(clean.clone() - ablated, head_zero_logits);
    assert!(
        attribution_error < TOLERANCE,
        "predicted versus observed ablation"
    );

    // 5. Restore a head ablation with its saved clean activation; this is not
    // clean-to-corrupted-input causal tracing. A wrong-head patch is a control.
    // Actions run in registration order: zero the head, then patch it.
    ablate.replace(head_output(0), head_zero_output);
    let restored = model.forward_with_hooks(input.clone(), Visibility::Causal, &mut ablate)?;
    let restoration_error = max_error(clean.clone(), restored);
    assert!(
        restoration_error < TOLERANCE,
        "clean activation restores logits"
    );
    let mut wrong_head = Hooks::new();
    wrong_head.replace(head_output(0), head_one_output);
    let wrong_patch =
        model.forward_with_hooks(input.clone(), Visibility::Causal, &mut wrong_head)?;
    let wrong_patch_effect = max_error(clean, wrong_patch);
    assert!(
        wrong_patch_effect > 0.1,
        "wrong-site patch must not restore logits"
    );

    // An additional API experiment, not a result claimed by the paper:
    // altering the future must affect position zero only with noncausal attention.
    let changed_future = input.clone().slice_assign(
        [0..1, 3..4, 0..4],
        Tensor::from_floats([[[5., -2., 3., 1.]]], &input.device()),
    );
    let mut visibility_effects = Vec::new();
    for visibility in [Visibility::Causal, Visibility::Bidirectional] {
        let original = model.forward_with_hooks(input.clone(), visibility, &mut Hooks::new())?;
        let ordinary_attention = model.attention.attend(
            model.attention.project_self(input.clone())?,
            None,
            None,
            matches!(visibility, Visibility::Causal),
        )?;
        parity_error = parity_error.max(max_error(
            original.clone(),
            model.readout(input.clone() + ordinary_attention),
        ));
        let changed =
            model.forward_with_hooks(changed_future.clone(), visibility, &mut Hooks::new())?;
        visibility_effects.push(max_error(original.narrow(1, 0, 1), changed.narrow(1, 0, 1)));
    }
    assert!(
        parity_error < TOLERANCE,
        "parity for both visibility policies"
    );
    assert!(visibility_effects[0] < TOLERANCE);
    assert!(visibility_effects[1] > 0.1);

    println!("Elhage et al. (2021): exact head decomposition on a fixed tiny transformer");
    println!("ordinary/instrumented max error: {parity_error:.8}");
    println!("additive decomposition max error: {decomposition_error:.8}");
    println!("head-zero ablation effect: {ablation_effect:.6}");
    println!("predicted/observed ablation max error: {attribution_error:.8}");
    println!("clean-patch restoration max error: {restoration_error:.8}");
    println!("wrong-head patch effect: {wrong_patch_effect:.6}");
    println!(
        "future-token effect, causal/bidirectional: {:.6} / {:.6}",
        visibility_effects[0], visibility_effects[1]
    );
    println!(
        "PASS: mathematical reproduction and intervention controls; no learned-circuit claim."
    );
    Ok(())
}
