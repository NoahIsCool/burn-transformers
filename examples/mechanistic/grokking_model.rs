//! The paper's one-block transformer, assembled from the crate's public attention.
//! Keeping this small recipe local makes the intervention boundaries easy to read.
#![allow(dead_code)] // Shared by separate training and checkpoint-analysis examples.
use burn::{
    backend::Flex,
    module::{Module, Param},
    tensor::{Distribution, Int, Tensor, TensorData, activation::relu, backend::Backend},
};
use safetensors::SafeTensors;
use sha2::{Digest, Sha256};
use transformers::{
    attention::{GroupedQueryAttention, GroupedQueryAttentionConfig},
    intervention::{Axis, Hooks, Point},
};

pub type B = Flex<f32, i32>;
pub const P: usize = 113;
pub const HIDDEN: usize = 128;
pub const NEURONS: usize = 512;

/// The final token's post-ReLU neurons, before the MLP output projection.
/// This recipe computes only that token, so the tensor is `[batch, neuron]`.
pub fn mlp_activation() -> Point<2> {
    Point::custom(
        "grokking.last_token.mlp.activation",
        [Axis::Batch, Axis::Channel],
    )
}

#[derive(Module, Debug)]
pub struct GrokkingTransformer<B: Backend = self::B> {
    embedding: Param<Tensor<B, 2>>,
    position: Param<Tensor<B, 2>>,
    attention: GroupedQueryAttention<B>,
    mlp_in: Param<Tensor<B, 2>>,
    mlp_in_bias: Param<Tensor<B, 2>>,
    mlp_out: Param<Tensor<B, 2>>,
    mlp_out_bias: Param<Tensor<B, 2>>,
    unembedding: Param<Tensor<B, 2>>,
}

impl GrokkingTransformer<B> {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path).map_err(|error| {
            std::io::Error::other(format!(
                "{path}: {error}; run python3 examples/mechanistic/grokking_prepare.py"
            ))
        })?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        if hash != "28f7212abd53937bc85ee0558536d89b0b882b3cfbec77d6942f605f7d2d4116" {
            return Err("expected the pinned epoch-40,000 checkpoint (SHA-256 mismatch)".into());
        }
        let tensors = SafeTensors::deserialize(&bytes)?;
        let device = Default::default();
        let load = |name: &str, shape: [usize; 2]| -> Tensor<B, 2> {
            let view = tensors.tensor(name).expect("pinned tensor name");
            let values: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            Tensor::from_data(TensorData::new(values, shape), &device)
        };
        let mut attention = GroupedQueryAttentionConfig::new(HIDDEN, 4, 4, 32).init(&device)?;
        // Authors store [head, head_channel, input]; flatten then transpose to
        // Burn's [input, output]. No normalization, tied weights, or Q/K biases.
        attention.q_proj.weight =
            Param::from_tensor(load("blocks.0.attn.W_Q", [HIDDEN, HIDDEN]).transpose());
        attention.k_proj.weight =
            Param::from_tensor(load("blocks.0.attn.W_K", [HIDDEN, HIDDEN]).transpose());
        attention.v_proj.weight =
            Param::from_tensor(load("blocks.0.attn.W_V", [HIDDEN, HIDDEN]).transpose());
        attention.o_proj.weight =
            Param::from_tensor(load("blocks.0.attn.W_O", [HIDDEN, HIDDEN]).transpose());
        Ok(Self {
            embedding: Param::from_tensor(load("embed.W_E", [HIDDEN, P + 1]).transpose()),
            position: Param::from_tensor(load("pos_embed.W_pos", [3, HIDDEN])),
            attention,
            mlp_in: Param::from_tensor(load("blocks.0.mlp.W_in", [NEURONS, HIDDEN]).transpose()),
            mlp_in_bias: Param::from_tensor(load("blocks.0.mlp.b_in", [1, NEURONS])),
            mlp_out: Param::from_tensor(load("blocks.0.mlp.W_out", [HIDDEN, NEURONS]).transpose()),
            mlp_out_bias: Param::from_tensor(load("blocks.0.mlp.b_out", [1, HIDDEN])),
            unembedding: Param::from_tensor(load("unembed.W_U", [HIDDEN, P + 1]).narrow(1, 0, P)),
        })
    }
}

impl<B: Backend> GrokkingTransformer<B> {
    /// Initialize the author's architecture from scratch. Every matrix except
    /// the readout uses N(0, 1/sqrt(128)); readout uses 1/sqrt(114).
    /// The full 114-class readout includes '=' during training, as in the source.
    pub fn init_random(device: &B::Device) -> transformers::Result<Self> {
        let normal = |shape| {
            Param::from_tensor(Tensor::random(
                shape,
                Distribution::Normal(0.0, 1.0 / (HIDDEN as f64).sqrt()),
                device,
            ))
        };
        let mut attention = GroupedQueryAttentionConfig::new(HIDDEN, 4, 4, 32).init(device)?;
        attention.q_proj.weight = normal([HIDDEN, HIDDEN]);
        attention.k_proj.weight = normal([HIDDEN, HIDDEN]);
        attention.v_proj.weight = normal([HIDDEN, HIDDEN]);
        attention.o_proj.weight = normal([HIDDEN, HIDDEN]);
        Ok(Self {
            embedding: normal([P + 1, HIDDEN]),
            position: normal([3, HIDDEN]),
            attention,
            mlp_in: normal([HIDDEN, NEURONS]),
            mlp_in_bias: Param::from_tensor(Tensor::zeros([1, NEURONS], device)),
            mlp_out: normal([NEURONS, HIDDEN]),
            mlp_out_bias: Param::from_tensor(Tensor::zeros([1, HIDDEN], device)),
            unembedding: Param::from_tensor(Tensor::random(
                [HIDDEN, P + 1],
                Distribution::Normal(0.0, 1.0 / ((P + 1) as f64).sqrt()),
                device,
            )),
        })
    }

    /// Ordinary differentiable forward. Hooks are available at the same boundary
    /// when training an auxiliary loss or intervening on the learned circuit.
    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> transformers::Result<Tensor<B, 2>> {
        self.forward_with_hooks(ids, &mut Hooks::new())
    }

    pub fn forward_with_hooks(
        &self,
        ids: Tensor<B, 2, Int>,
        hooks: &mut Hooks<'_, B>,
    ) -> transformers::Result<Tensor<B, 2>> {
        let (residual, activation) = self.encode_ids(ids)?;
        self.continue_with_hooks(residual, activation, hooks)
    }

    /// Capture the final-token residual and post-ReLU neurons, before readout.
    pub fn encode(
        &self,
        start: usize,
        count: usize,
    ) -> transformers::Result<(Tensor<B, 2>, Tensor<B, 2>)> {
        let pairs: Vec<usize> = (start..start + count).collect();
        self.encode_pairs(&pairs)
    }

    /// Encode arbitrary dataset rows. The same differentiable graph serves
    /// training, validation, and the released-checkpoint interpretation.
    pub fn encode_pairs(
        &self,
        pairs: &[usize],
    ) -> transformers::Result<(Tensor<B, 2>, Tensor<B, 2>)> {
        let device = self.embedding.val().device();
        let count = pairs.len();
        let ids: Vec<i32> = pairs
            .iter()
            .flat_map(|&pair| [(pair / P) as i32, (pair % P) as i32, P as i32])
            .collect();
        let ids = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [count, 3]), &device);
        self.encode_ids(ids)
    }

    pub fn encode_ids(
        &self,
        ids: Tensor<B, 2, Int>,
    ) -> transformers::Result<(Tensor<B, 2>, Tensor<B, 2>)> {
        let [count, tokens] = ids.dims();
        assert_eq!(tokens, 3, "modular-addition inputs are [a, b, =]");
        let ids = ids.reshape([count * 3]);
        let hidden = self
            .embedding
            .val()
            .select(0, ids)
            .reshape([count, 3, HIDDEN])
            + self.position.val().unsqueeze::<3>();
        // Only the '=' query is needed in a single-layer model. Its causal row
        // can attend to all three positions, so no element in this row is masked.
        let last = hidden.clone().narrow(1, 2, 1);
        let projection = self.attention.project(last.clone(), hidden)?;
        let attended = self.attention.attend(projection, None, None, false)?;
        let residual = (last + attended).reshape([count, HIDDEN]);
        let activation = relu(residual.clone().matmul(self.mlp_in.val()) + self.mlp_in_bias.val());
        Ok((residual, activation))
    }

    /// Resume the ordinary readout, exposing the neurons through public hooks.
    /// Custom recipes declare their available points and apply them at the
    /// corresponding computation boundary; selection and validation are shared.
    pub fn continue_with_hooks(
        &self,
        residual: Tensor<B, 2>,
        activation: Tensor<B, 2>,
        hooks: &mut Hooks<'_, B>,
    ) -> transformers::Result<Tensor<B, 2>> {
        hooks.run(vec![mlp_activation().info()], |hooks| {
            let activation = hooks.apply(mlp_activation(), activation)?;
            Ok(
                (residual + activation.matmul(self.mlp_out.val()) + self.mlp_out_bias.val())
                    .matmul(self.unembedding.val()),
            )
        })
    }

    /// Embedding [113,128] and neuron-to-logit [512,113] for Fourier progress.
    pub fn fourier_weights(&self) -> (Vec<f32>, Vec<f32>) {
        (
            self.embedding
                .val()
                .narrow(0, 0, P)
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            self.mlp_out
                .val()
                .matmul(self.unembedding.val().narrow(1, 0, P))
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
        )
    }

    /// Return cosine/sine readout directions as columns in the 512D latent space.
    pub fn fourier_directions(&self, frequencies: &[usize]) -> Vec<Vec<f64>> {
        let neuron_to_logit = self
            .mlp_out
            .val()
            .matmul(self.unembedding.val().narrow(1, 0, P))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        frequencies
            .iter()
            .flat_map(|&frequency| [false, true].into_iter().map(move |sine| (frequency, sine)))
            .map(|(frequency, sine)| {
                (0..NEURONS)
                    .map(|neuron| {
                        (0..P)
                            .map(|output| {
                                let phase =
                                    std::f64::consts::TAU * (frequency * output) as f64 / P as f64;
                                neuron_to_logit[neuron * P + output] as f64
                                    * if sine { phase.sin() } else { phase.cos() }
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}
