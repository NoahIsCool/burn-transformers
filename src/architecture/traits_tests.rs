//! Experiment-trait tests: generic caching and patching code over a small staged model.

use burn::{
    backend::Flex,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{Int, Tensor},
};

use super::*;
use crate::{
    experiment::{ExecutionContext, HiddenTransform, ResidualBlock},
    intervention::{DynamicActivation, Point},
};

/// Deterministic F32 CPU backend.
type B = Flex<f32, i32>;

/// A decoder of tokenwise residual blocks with a fixed final scaling as "normalization".
#[derive(burn::module::Module, Debug)]
struct TinyStack<B: Backend> {
    /// Token embedding table `[vocabulary, width]`.
    embedding: Embedding<B>,
    /// Residual blocks with linear mixers and feed-forwards.
    blocks: Vec<ResidualBlock<B, Linear<B>, Linear<B>>>,
}

impl TinyStack<B> {
    /// Builds a seeded random stack of `layers` blocks over `width` channels.
    fn new(layers: usize, width: usize) -> Self {
        let device = Default::default();
        B::seed(&device, 3);
        let block = || {
            ResidualBlock::without_normalization(
                width,
                LinearConfig::new(width, width).init(&device),
                LinearConfig::new(width, width).init(&device),
            )
            .unwrap()
        };
        Self {
            embedding: EmbeddingConfig::new(16, width).init(&device),
            blocks: (0..layers).map(|_| block()).collect(),
        }
    }
}

/// Stand-in final normalization: a fixed halving, enough to make the stage observable.
fn final_scale(hidden: Tensor<B, 3>) -> Tensor<B, 3> {
    hidden * 0.5
}

impl HookedModel<B> for TinyStack<B> {
    type Input = CausalLmInput<B>;
    type Output = Tensor<B, 3>;

    /// Embedding, every block's residual and branch points, then the final norm.
    fn activation_points(&self) -> Result<Vec<PointInfo>> {
        let mut points = vec![Stack::Decoder.embedding().info()];
        for index in 0..self.blocks.len() {
            points.extend(self.blocks[index].activation_points(self.layer(index)));
        }
        points.push(Stack::Decoder.final_norm().info());
        Ok(points)
    }

    /// Runs embedding, blocks, and the final scaling inside one scope.
    fn forward_with_hooks(
        &self,
        input: CausalLmInput<B>,
        _options: &ForwardOptions,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, tokens] = input.token_ids.dims();
        let context = ExecutionContext::contiguous(batch, tokens, &input.token_ids.device())?;
        hooks.run(HookedModel::activation_points(self)?, |hooks| {
            let mut hidden = hooks.apply(
                Stack::Decoder.embedding(),
                self.embedding.forward(input.token_ids),
            )?;
            for (index, block) in self.blocks.iter().enumerate() {
                hidden = block.forward(hidden, &context, self.layer(index), hooks)?;
            }
            hooks.apply(Stack::Decoder.final_norm(), final_scale(hidden))
        })
    }
}

impl StagedModel<B> for TinyStack<B> {
    /// Number of residual blocks.
    fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// The fixture is decoder-only.
    fn stack(&self) -> Stack {
        Stack::Decoder
    }

    /// Embeds tokens with contiguous positions in an embedding-only scope.
    fn embed_with_hooks(
        &self,
        input: CausalLmInput<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        let [batch, tokens] = input.token_ids.dims();
        let context = ExecutionContext::contiguous(batch, tokens, &input.token_ids.device())?;
        let hidden = hooks.run(vec![Stack::Decoder.embedding().info()], |hooks| {
            hooks.apply(
                Stack::Decoder.embedding(),
                self.embedding.forward(input.token_ids),
            )
        })?;
        HiddenState::new(hidden, context)
    }

    /// Runs one validated block range inside a scope of exactly those blocks' points.
    fn run_layers_with_hooks(
        &self,
        mut state: HiddenState<B>,
        range: Range<usize>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        state.validate_range(range.clone(), self.blocks.len())?;
        state.validate(state.hidden.dims()[2])?;
        let points = range
            .clone()
            .flat_map(|index| self.blocks[index].activation_points(self.layer(index)))
            .collect();
        let context = state.context.clone();
        state.hidden = hooks.run(points, |hooks| {
            let mut hidden = state.hidden.clone();
            for index in range.clone() {
                hidden = self.blocks[index].forward(hidden, &context, self.layer(index), hooks)?;
            }
            Ok(hidden)
        })?;
        state.next_layer = range.end;
        Ok(state)
    }

    /// Applies the final scaling once, after every block has run.
    fn normalize_with_hooks(
        &self,
        mut state: HiddenState<B>,
        hooks: &mut Hooks<'_, B>,
    ) -> Result<HiddenState<B>> {
        state.validate_range(self.blocks.len()..self.blocks.len(), self.blocks.len())?;
        state.hidden = hooks.run(vec![Stack::Decoder.final_norm().info()], |hooks| {
            hooks.apply(
                Stack::Decoder.final_norm(),
                final_scale(state.hidden.clone()),
            )
        })?;
        state.normalized = true;
        Ok(state)
    }
}

/// Every captured point of one run, in catalog order.
type ActivationCache<B> = Vec<(PointInfo, DynamicActivation<B>)>;

/// The trait-level cache example: capture every listed point for any hooked model.
fn run_with_cache<B: Backend, M: HookedModel<B>>(
    model: &M,
    input: M::Input,
) -> Result<(M::Output, ActivationCache<B>)> {
    let mut hooks = Hooks::new();
    let handles = model
        .activation_points()?
        .into_iter()
        .map(|point| hooks.capture_dynamic(point))
        .collect::<Vec<_>>();
    let output = model.forward_with_hooks(input, &ForwardOptions::default(), &mut hooks)?;
    let cache = handles
        .iter()
        .map(|handle| Ok((handle.point().clone(), hooks.get_dynamic(handle)?)))
        .collect::<Result<_>>()?;
    Ok((output, cache))
}

/// The trait-level patching example: replace the residual entering one layer.
fn patch_residual<B: Backend, M: StagedModel<B>>(
    model: &M,
    input: CausalLmInput<B>,
    donor: Tensor<B, 3>,
    layer: usize,
) -> Result<Tensor<B, 3>> {
    let state = model.embed(input)?;
    let mut state = model.run_layers(state, 0..layer)?;
    state.hidden = donor;
    let state = model.run_layers(state, layer..model.num_layers())?;
    Ok(model.normalize(state)?.hidden)
}

/// Token inputs without padding or explicit positions.
fn tokens(ids: [[i32; 3]; 1]) -> CausalLmInput<B> {
    CausalLmInput {
        token_ids: Tensor::<B, 2, Int>::from_ints(ids, &Default::default()),
        token_mask: None,
        position_ids: None,
    }
}

/// Flattens a rank-three tensor for exact comparison.
fn values(tensor: Tensor<B, 3>) -> Vec<f32> {
    tensor.into_data().to_vec().unwrap()
}

/// Generic caching sees every listed point with its emitted shape and matches the output.
#[test]
fn generic_activation_cache_captures_every_listed_point() {
    let model = TinyStack::new(2, 4);
    let (output, cache) = run_with_cache(&model, tokens([[1, 2, 3]])).unwrap();
    let points = HookedModel::activation_points(&model).unwrap();
    assert_eq!(cache.len(), points.len());
    for ((point, activation), expected) in cache.iter().zip(&points) {
        assert_eq!(point, expected);
        assert_eq!(activation.shape, [1, 3, 4], "{}", point.name);
    }
    let (_, final_norm) = cache.last().unwrap();
    assert_eq!(
        values(final_norm.clone().into_tensor::<3>().unwrap()),
        values(output)
    );
}

/// Staged execution reproduces the single-scope forward pass exactly.
#[test]
fn staged_execution_matches_the_hooked_forward_pass() {
    let model = TinyStack::new(3, 4);
    let forward = model
        .forward_with_hooks(
            tokens([[4, 5, 6]]),
            &ForwardOptions::default(),
            &mut Hooks::new(),
        )
        .unwrap();
    let staged = model.run_stack(tokens([[4, 5, 6]])).unwrap();
    assert!(staged.normalized);
    assert_eq!(staged.next_layer, 3);
    assert_eq!(values(staged.hidden), values(forward));
    assert_eq!(model.layer(2), Layer::decoder(2));
    assert_eq!(Stack::Encoder.layer(1), Layer::encoder(1));
}

/// Generic residual patching: a clean donor restores the clean output.
#[test]
fn generic_patching_restores_the_clean_run() {
    let model = TinyStack::new(3, 4);
    let clean = tokens([[1, 2, 3]]);
    let corrupted = tokens([[7, 2, 3]]);
    let clean_output = values(model.run_stack(clean.clone()).unwrap().hidden);
    let corrupted_output = values(model.run_stack(corrupted.clone()).unwrap().hidden);
    assert_ne!(clean_output, corrupted_output);
    for layer in 0..=model.num_layers() {
        let donor = model
            .run_layers(model.embed(clean.clone()).unwrap(), 0..layer)
            .unwrap()
            .hidden;
        let patched = patch_residual(&model, corrupted.clone(), donor, layer).unwrap();
        assert_eq!(values(patched), clean_output, "layer {layer}");
    }
}

/// Stage scopes reject points outside the stage, and stages enforce block order.
#[test]
fn stages_are_separate_scopes_with_ordered_ranges() {
    let model = TinyStack::new(2, 4);
    let state = model.embed(tokens([[1, 2, 3]])).unwrap();
    let mut hooks = Hooks::new();
    hooks.capture(Layer::decoder(1).residual_post());
    let error = model
        .run_layers_with_hooks(state.clone(), 0..1, &mut hooks)
        .unwrap_err();
    assert!(error.to_string().contains("unavailable"), "{error}");
    assert!(model.run_layers(state.clone(), 1..2).is_err());
    let mut hooks = Hooks::new();
    let residual = Point::<3>::try_from(model.layer(1).residual_post().info()).unwrap();
    let captured = hooks.capture(residual);
    let state = model.run_layers(state, 0..1).unwrap();
    model
        .run_layers_with_hooks(state, 1..2, &mut hooks)
        .unwrap();
    assert_eq!(hooks.get(&captured).unwrap().dims(), [1, 3, 4]);
}
