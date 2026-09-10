use std::{path::Path, time::Instant};
use burn::{backend::Flex, tensor::{Int, Tensor}};
use transformers::{CausalLmInput, checkpoint::{CheckpointManifest, DTypePolicy, LoadOptions, TensorDType, execute_local_load_plan}, intervention::{Hooks, Layer}, models::qwen3_5::{Qwen3_5BackboneRecipe, Qwen3_5RecipeLoadOptions, Qwen3_5CheckpointOptions, Qwen3_5CheckpointSource, Qwen3_5ForCausalLm}};

type B = Flex<f32, i32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root_string = std::env::args().nth(1).ok_or("expected model directory")?;
    let root = Path::new(&root_string);
    let device = Default::default();
    let options = LoadOptions { dtype_policy: DTypePolicy::Cast(TensorDType::F32), ..Default::default() };
    let recipe = Qwen3_5BackboneRecipe::from_local_config(root)?.prefix(2)?;
    let start = Instant::now();
    let loaded = recipe.load_pretrained::<B>(root, &Qwen3_5RecipeLoadOptions { checkpoint: options.clone(), ..Default::default() }, &device)?;
    let prefix_load_seconds = start.elapsed().as_secs_f64();
    let input = CausalLmInput { token_ids: Tensor::<B, 2, Int>::from_ints([[1, 2, 3]], &device), token_mask: None, position_ids: None };
    let mut hooks = Hooks::new();
    let captured = hooks.capture(Layer::decoder(1).mlp_activation().token(2));
    let start = Instant::now();
    let embedded = loaded.model.embed(&input)?;
    let prefix = loaded.model.run_layers_with_hooks(embedded, 0..2, &mut hooks)?;
    let prefix = loaded.model.normalize(prefix)?;
    let capture = hooks.get(&captured)?;
    let capture_values = capture.to_data().to_vec::<f32>()?;
    let prefix_values = prefix.hidden.to_data().to_vec::<f32>()?;
    let prefix_forward_seconds = start.elapsed().as_secs_f64();
    assert!(prefix_values.iter().chain(&capture_values).all(|x| x.is_finite()));

    let manifest = CheckpointManifest::discover(root, &options)?;
    let source = if manifest.tensor("model.language_model.embed_tokens.weight").is_some() { Qwen3_5CheckpointSource::Multimodal } else { Qwen3_5CheckpointSource::TextOnly };
    let config = recipe.source_config().clone();
    let contract = config.checkpoint_contract(Qwen3_5CheckpointOptions { source, include_mtp: false })?;
    let plan = contract.build_load_plan(&manifest, &options)?;
    let start = Instant::now();
    let mut full = Qwen3_5ForCausalLm::<B>::init(config, &device)?;
    let full_load = execute_local_load_plan(&manifest, &plan, &mut full)?;
    let full_load_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let embedded_full = full.model.embed(&input)?;
    let mut reference_hidden = embedded_full.hidden;
    for index in 0..2 {
        reference_hidden = full.model.layers[index].forward(reference_hidden, embedded_full.context.positions.clone(), embedded_full.context.token_mask.as_ref(), full.model.layer_rope[index], embedded_full.context.evaluated_context)?;
    }
    reference_hidden = full.model.norm.forward(reference_hidden);
    let full_values = reference_hidden.to_data().to_vec::<f32>()?;
    let full_prefix_forward_seconds = start.elapsed().as_secs_f64();
    let max_abs_error = prefix_values.iter().zip(&full_values).map(|(a,b)| (*a-*b).abs()).fold(0.0_f32, f32::max);
    assert_eq!(prefix_values, full_values);
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "scope": "Local pinned-checkpoint selective-load and hidden-state parity smoke; no new support, task-quality, or performance qualification",
        "backend": "Flex CPU", "token_ids": [[1,2,3]], "blocks": [0,1],
        "source_payload_bytes": loaded.checkpoint.stats.source_payload_bytes,
        "total_checkpoint_payload_bytes": manifest.tensor_bytes,
        "prefix_report": loaded.report,
        "capture_shape": capture.dims(), "capture_dtype": format!("{:?}", capture.dtype()),
        "hidden_shape": prefix.hidden.dims(), "hidden_dtype": format!("{:?}", prefix.hidden.dtype()),
        "all_captured_and_hidden_values_finite": true,
        "prefix_load_seconds": prefix_load_seconds, "prefix_forward_seconds": prefix_forward_seconds,
        "full_model_load_seconds": full_load_seconds, "full_model_source_payload_bytes": full_load.stats.source_payload_bytes,
        "full_model_prefix_forward_seconds": full_prefix_forward_seconds,
        "comparison": "Same first two individually executed blocks from the full text model, followed by its final normalization",
        "compared_hidden_values": prefix_values.len(), "hidden_max_absolute_error": max_abs_error, "hidden_exact_equal": true
    }))?);
    Ok(())
}
