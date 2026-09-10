//! The authors' one-block recipe, using shared attention and public interventions.
//!
//! Two tokens `[a,b]` modulo 59 enter a width-128 residual stream with four
//! width-32 attention heads and a biased 512-neuron ReLU MLP. Attention is
//! unmasked: `A = alpha * softmax(Q K^T / sqrt(32)) + (1-alpha) J`. Every
//! entry of J is one, so alpha=0 sums values rather than averaging them.
//! The final token predicts `[batch,59]` logits without normalization or dropout.
//!
//! The reference path uses F32 tensors and eager F32 softmax. Token-embedding
//! interventions run before position addition, preserving the paper’s PCA
//! interpretation. Registered parameters are shared by inference and training;
//! this fixed two-token graph has no decode cache.
use std::{collections::BTreeMap, path::Path};

use burn::{
    backend::Flex,
    module::{Module, Param},
    tensor::{Int, Tensor, TensorData, activation::relu, backend::Backend},
};
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use transformers::{
    attention::{AttentionPrecisionMode, GroupedQueryAttention, GroupedQueryAttentionConfig},
    intervention::{Axis, Hooks, Layer, Point, PointInfo},
};

/// Deterministic F32 CPU backend used to replay prepared reference fixtures.
pub type B = Flex<f32, i32>;
/// Prime modulus and number of token/output classes.
pub const P: usize = 59;
/// Residual-stream width of the paper’s single transformer block.
pub const HIDDEN: usize = 128;
/// Width of the biased ReLU MLP’s intermediate activation.
pub const NEURONS: usize = 512;

/// Token lookup output BEFORE position embeddings, `[batch, token, channel]`.
/// Editing this point is equivalent to changing the token embedding table alone.
pub fn token_embedding() -> Point<3> {
    Point::custom(
        "clock_pizza.token_embedding",
        [Axis::Batch, Axis::Token, Axis::Channel],
    )
}

/// Prepared pinned model, embedding PCA basis, and independent reference logits.
pub struct Fixture {
    /// Transformer assembled from the named released checkpoint.
    pub model: Transformer,
    /// Embedding mean `[1,128]`, restored after each centered PCA projection.
    pub mean: Tensor<B, 2>,
    /// Principal directions as columns, in descending explained-variance order.
    pub components: Tensor<B, 2>,
    /// Flattened `[59*59,59]` F32 reference logits for each fixed intervention.
    pub references: BTreeMap<String, Vec<f32>>,
}

impl Fixture {
    /// Load a prepared Clock or Pizza fixture after checking its manifest SHA-256.
    /// Reject wrong tensor shapes, non-F32 storage, and nonfinite values. Q/K/V
    /// weights are flattened from `[head, head_channel, input]` and transposed
    /// to Burn’s `[input, output]`; embedding and MLP matrices are mapped likewise.
    pub fn load(directory: &Path, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let path = directory.join(format!("{name}.safetensors"));
        let bytes = std::fs::read(&path).map_err(|error| {
            std::io::Error::other(format!(
                "{}: {error}; run python3 examples/mechanistic/clock_and_pizza/prepare.py",
                path.display()
            ))
        })?;
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("reference.json"))?)?;
        let expected = metadata["outputs"][name]["sha256"]
            .as_str()
            .ok_or("reference.json is missing the prepared artifact SHA-256")?;
        if format!("{:x}", Sha256::digest(&bytes)) != expected {
            return Err(format!(
                "{}: prepared artifact SHA-256 mismatch; rerun prepare.py --offline --refresh",
                path.display()
            )
            .into());
        }
        let tensors = SafeTensors::deserialize(&bytes)?;
        let values =
            |key: &str, expected: &[usize]| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                let view = tensors.tensor(key)?;
                if view.dtype() != Dtype::F32 || view.shape() != expected {
                    return Err(format!(
                        "{key}: expected F32 {expected:?}, got {:?} {:?}",
                        view.dtype(),
                        view.shape()
                    )
                    .into());
                }
                let values: Vec<_> = view
                    .data()
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect();
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(format!("{key}: all values must be finite").into());
                }
                Ok(values)
            };
        let device = Default::default();
        let load = |key: &str,
                    stored: &[usize],
                    shape: [usize; 2]|
         -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
            Ok(Tensor::from_data(
                TensorData::new(values(key, stored)?, shape),
                &device,
            ))
        };
        let mut attention = GroupedQueryAttentionConfig::new(HIDDEN, 4, 4, 32)
            .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax)
            .init(&device)?;
        // Authors store [head, head channel, input]; Burn stores [input, output].
        attention.q_proj.weight = Param::from_tensor(
            load("blocks.0.attn.W_Q", &[4, 32, HIDDEN], [HIDDEN, HIDDEN])?.transpose(),
        );
        attention.k_proj.weight = Param::from_tensor(
            load("blocks.0.attn.W_K", &[4, 32, HIDDEN], [HIDDEN, HIDDEN])?.transpose(),
        );
        attention.v_proj.weight = Param::from_tensor(
            load("blocks.0.attn.W_V", &[4, 32, HIDDEN], [HIDDEN, HIDDEN])?.transpose(),
        );
        attention.o_proj.weight = Param::from_tensor(
            load("blocks.0.attn.W_O", &[HIDDEN, HIDDEN], [HIDDEN, HIDDEN])?.transpose(),
        );
        let alpha = match name {
            "pizza" => 0.0,
            "clock" => 1.0,
            _ => return Err(format!("unknown representative checkpoint {name}").into()),
        };
        let model = Transformer {
            embedding: Param::from_tensor(
                load("embed.W_E", &[HIDDEN, P], [HIDDEN, P])?.transpose(),
            ),
            position: Param::from_tensor(load("pos_embed.W_pos", &[2, HIDDEN], [2, HIDDEN])?),
            attention,
            alpha,
            mlp_in: Param::from_tensor(
                load("blocks.0.mlp.W_in", &[NEURONS, HIDDEN], [NEURONS, HIDDEN])?.transpose(),
            ),
            mlp_in_bias: Param::from_tensor(load("blocks.0.mlp.b_in", &[NEURONS], [1, NEURONS])?),
            mlp_out: Param::from_tensor(
                load("blocks.0.mlp.W_out", &[HIDDEN, NEURONS], [HIDDEN, NEURONS])?.transpose(),
            ),
            mlp_out_bias: Param::from_tensor(load("blocks.0.mlp.b_out", &[HIDDEN], [1, HIDDEN])?),
            unembedding: Param::from_tensor(load("unembed.W_U", &[HIDDEN, P], [HIDDEN, P])?),
        };
        let references = ["baseline", "first2", "first6", "circle2", "circle3"]
            .into_iter()
            .map(|case| {
                Ok((
                    case.into(),
                    values(&format!("reference.{case}"), &[P * P, P])?,
                ))
            })
            .collect::<Result<_, Box<dyn std::error::Error>>>()?;
        Ok(Self {
            model,
            mean: load("pca.mean", &[1, HIDDEN], [1, HIDDEN])?,
            components: load("pca.components", &[HIDDEN, 20], [HIDDEN, 20])?,
            references,
        })
    }
}

#[derive(Module, Debug)]
/// Registered one-block module exposing embedding, attention, MLP, and residual hooks.
pub struct Transformer<B: Backend = burn::backend::Flex<f32, i32>> {
    /// Token lookup table `[59,128]`, also exported for learned-geometry analysis.
    pub embedding: Param<Tensor<B, 2>>,
    position: Param<Tensor<B, 2>>,
    attention: GroupedQueryAttention<B>,
    /// Attention mixture coefficient in `[0,1]`; zero selects unnormalized summation.
    pub alpha: f64,
    mlp_in: Param<Tensor<B, 2>>,
    mlp_in_bias: Param<Tensor<B, 2>>,
    mlp_out: Param<Tensor<B, 2>>,
    mlp_out_bias: Param<Tensor<B, 2>>,
    unembedding: Param<Tensor<B, 2>>,
}

impl<B: Backend> Transformer<B> {
    /// Initialize the authors' Gaussian weight scales with backend-controlled RNG.
    /// Standard deviations are reciprocal square roots of the documented width
    /// for each matrix; biases start at zero. Alpha must lie in `[0,1]`.
    #[allow(dead_code)]
    pub fn init_random(alpha: f64, device: &B::Device) -> transformers::Result<Self> {
        use burn::tensor::Distribution;
        assert!((0.0..=1.0).contains(&alpha));
        let normal = |shape: [usize; 2], denominator: usize| {
            Param::from_tensor(Tensor::random(
                shape,
                Distribution::Normal(0.0, 1.0 / (denominator as f64).sqrt()),
                device,
            ))
        };
        let mut attention = GroupedQueryAttentionConfig::new(HIDDEN, 4, 4, 32)
            .with_precision_mode(AttentionPrecisionMode::EagerFp32Softmax)
            .init(device)?;
        attention.q_proj.weight = normal([HIDDEN, HIDDEN], HIDDEN);
        attention.k_proj.weight = normal([HIDDEN, HIDDEN], HIDDEN);
        attention.v_proj.weight = normal([HIDDEN, HIDDEN], HIDDEN);
        attention.o_proj.weight = normal([HIDDEN, HIDDEN], HIDDEN);
        Ok(Self {
            embedding: normal([P, HIDDEN], HIDDEN),
            position: normal([2, HIDDEN], HIDDEN),
            attention,
            alpha,
            mlp_in: normal([HIDDEN, NEURONS], NEURONS),
            mlp_in_bias: Param::from_tensor(Tensor::zeros([1, NEURONS], device)),
            mlp_out: normal([NEURONS, HIDDEN], HIDDEN),
            mlp_out_bias: Param::from_tensor(Tensor::zeros([1, HIDDEN], device)),
            unembedding: normal([HIDDEN, P], P),
        })
    }

    /// Declare selectable token-embedding and single-block intervention boundaries.
    /// Shared attention points retain their batch/head/query/key axis contracts.
    pub fn activation_points(&self) -> Vec<PointInfo> {
        let layer = Layer::decoder(0);
        let mut points = vec![token_embedding().info(), layer.residual_pre().info()];
        points.extend(layer.self_attention().all());
        points.extend([
            layer.attention_output().info(),
            layer.mlp_input().info(),
            layer.mlp_pre_activation().info(),
            layer.mlp_activation().info(),
            layer.mlp_output().info(),
            layer.residual_post().info(),
        ]);
        points
    }

    /// Evaluate consecutive `(a,b)` pairs, returning final-token `[batch,59]` logits.
    /// Flattened grid indices use `pair = a*59+b` and must stay within the grid.
    pub fn forward_with_hooks(
        &self,
        start: usize,
        count: usize,
        hooks: &mut Hooks<'_, B>,
    ) -> transformers::Result<Tensor<B, 2>> {
        let ids: Vec<i32> = (start..start + count)
            .flat_map(|pair| [(pair / P) as i32, (pair % P) as i32])
            .collect();
        let ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ids, [count, 2]),
            &self.embedding.device(),
        );
        self.forward_ids_with_hooks(ids, hooks)
    }

    /// Train or evaluate `[batch,2]` integer pairs through the same hooked graph.
    /// Tokens have full bidirectional visibility, with no padding or cache. Hooks
    /// may edit the unnormalized attention mixture or `[batch,2,512]` MLP neurons;
    /// the second token’s final residual supplies `[batch,59]` output logits.
    pub fn forward_ids_with_hooks(
        &self,
        ids: Tensor<B, 2, Int>,
        hooks: &mut Hooks<'_, B>,
    ) -> transformers::Result<Tensor<B, 2>> {
        let [count, tokens] = ids.dims();
        assert_eq!(tokens, 2, "modular addition uses two input tokens");
        hooks.run(self.activation_points(), |hooks| {
            let ids = ids.reshape([count * 2]);
            let tokens = self
                .embedding
                .val()
                .select(0, ids)
                .reshape([count, 2, HIDDEN]);
            let tokens = hooks.apply(token_embedding(), tokens)?;
            let layer = Layer::decoder(0);
            let hidden = hooks.apply(
                layer.residual_pre(),
                tokens + self.position.val().unsqueeze::<3>(),
            )?;
            let projection = self.attention.project_self(hidden.clone())?;
            // Both models are unmasked. J has entries ONE: alpha=0 sums values,
            // and is deliberately not uniform probability attention (entries 1/2).
            let attended = self.attention.attend_with_pattern(
                projection,
                None,
                None,
                false,
                layer.self_attention(),
                hooks,
                |pattern| Ok(pattern * self.alpha + (1.0 - self.alpha)),
            )?;
            let residual = hidden + hooks.apply(layer.attention_output(), attended)?;
            let input = hooks.apply(layer.mlp_input(), residual.clone())?;
            let pre = input
                .reshape([count * 2, HIDDEN])
                .matmul(self.mlp_in.val())
                .reshape([count, 2, NEURONS])
                + self.mlp_in_bias.val().unsqueeze::<3>();
            let pre = hooks.apply(layer.mlp_pre_activation(), pre)?;
            let activation = hooks.apply(layer.mlp_activation(), relu(pre))?;
            let output = activation
                .reshape([count * 2, NEURONS])
                .matmul(self.mlp_out.val())
                .reshape([count, 2, HIDDEN])
                + self.mlp_out_bias.val().unsqueeze::<3>();
            let residual = residual + hooks.apply(layer.mlp_output(), output)?;
            let residual = hooks.apply(layer.residual_post(), residual)?;
            Ok(residual
                .narrow(1, 1, 1)
                .reshape([count, HIDDEN])
                .matmul(self.unembedding.val()))
        })
    }
}
