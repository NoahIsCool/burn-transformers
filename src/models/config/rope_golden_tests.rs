//! Golden parity between the shared RoPE translator and Hugging Face Transformers.
//!
//! Every case in `rope_goldens.json` records the raw serialized RoPE metadata
//! of a config (flat, legacy `rope_scaling`/`type`, or per-layer-type), the
//! surrounding fields passed to [`RopeContext`], and the inverse frequencies
//! and `attention_scaling` that the Transformers reference computes for it.
//! The test translates the same metadata with [`RopeParameterSet::to_rope_config`],
//! evaluates [`RotaryFrequencies`] at the recorded context, and requires a
//! relative frequency error of at most `1e-6` (the reference evaluates in F32,
//! this crate in F64 rounded once to F32) and an amplitude error of `1e-12`.
//!
//! Coverage: default, legacy linear with top-level theta, dynamic NTK inside
//! and beyond its context (including a partial rotary width), YaRN defaults,
//! betas with an explicit `attention_factor`, the `mscale` ratio, untruncated
//! partial YaRN, DeepSeek-V2-Lite under both mscale conventions, LongRoPE
//! short/long tables (nested and Phi-3 top-level original context), Phi-3
//! top-level partial RoPE, Llama 3 bands, the pinned Qwen3.5-0.8B and
//! Qwen3-30B-A3B fixtures, the tiny goldens `llama3`, `gemma3-text` (both
//! layer types), and `gpt-oss` (YaRN, `truncate: false`), and Gemma-4-style
//! per-layer proportional RoPE.
//!
//! References: Transformers 5.2.0 with torch 2.10.0 (the local install), each
//! case through a real config class so Transformers' own standardization and
//! precedence run; proportional RoPE through Transformers 5.8.0
//! `modeling_rope_utils.py` loaded standalone, because 5.2 lacks it; and the
//! DeepSeek reference amplitude transcribed from DeepSeek-V2-Lite
//! `modeling_deepseek.py` at revision `604d5664dddd88a0433dbae533b7fe9472482de0`
//! (its YaRN frequencies equal Transformers' truncated YaRN).
//!
//! Regenerate from the repository root with the script below; the second
//! argument is an unpacked Transformers 5.8.0 wheel (here a `uv` cache entry):
//!
//! ```text
//! script=$(mktemp -d)/gen_rope_goldens.py
//! sed -n '/^\/\/! ```python$/,/^\/\/! ```$/p' src/models/config/rope_golden_tests.rs \
//!   | sed -e '1d' -e '$d' -e 's|^//! \{0,1\}||' > "$script"
//! python3 -I "$script" "$PWD" ~/.cache/uv/archive-v0/yCiHBh6FcGFsfD1TNcf9J \
//!   > src/models/config/rope_goldens.json
//! ```
//!
//! ```python
//! import copy, importlib.util, json, math, sys, types
//!
//! import torch
//! import transformers
//! from transformers import AutoConfig, PretrainedConfig
//! from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
//! from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
//! from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RotaryEmbedding
//! from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
//! from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
//! from transformers.models.llama.configuration_llama import LlamaConfig
//! from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
//! from transformers.models.phi3.configuration_phi3 import Phi3Config
//! from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding
//! from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
//! from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
//! from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
//! from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding
//!
//! assert transformers.__version__ == "5.2.0", transformers.__version__
//! REPO, HF58 = sys.argv[1], sys.argv[2]
//!
//! # Transformers 5.8.0 `modeling_rope_utils.py` loaded standalone (5.2 lacks
//! # proportional RoPE); its only package import is `.utils`, served by 5.2.
//! package = types.ModuleType("hf58")
//! package.__path__ = []
//! sys.modules["hf58"] = package
//! sys.modules["hf58.utils"] = transformers.utils
//! spec = importlib.util.spec_from_file_location("hf58.modeling_rope_utils", f"{HF58}/transformers/modeling_rope_utils.py")
//! rope58 = importlib.util.module_from_spec(spec)
//! spec.loader.exec_module(rope58)
//!
//! cases = []
//!
//!
//! def record(name, reference, rope_parameters, context, evaluated_context, inv_freq, attention, layer_type=None):
//!     cases.append(
//!         {
//!             "name": name,
//!             "reference": reference,
//!             "rope_parameters": rope_parameters,
//!             "layer_type": layer_type,
//!             "context": context,
//!             "evaluated_context": evaluated_context,
//!             "inv_freq": [float(v) for v in inv_freq.tolist()],
//!             "attention_scaling": float(attention),
//!         }
//!     )
//!
//!
//! def llama(params=None, **kwargs):
//!     base = dict(hidden_size=64, num_attention_heads=4, head_dim=16, max_position_embeddings=128)
//!     base.update(kwargs)
//!     if params is not None:
//!         base["rope_parameters"] = copy.deepcopy(params)
//!     return LlamaConfig(**base)
//!
//!
//! def from_embedding(name, params, config, embedding, context, evaluated_context, reference):
//!     record(name, reference, params, context, evaluated_context, embedding.inv_freq, embedding.attention_scaling)
//!
//!
//! def from_init(name, params, config, context, evaluated_context, seq_len=None, reference="", layer_type=None):
//!     rope_type = config.rope_parameters[layer_type]["rope_type"] if layer_type else config.rope_parameters["rope_type"]
//!     kwargs = {"layer_type": layer_type} if layer_type else {}
//!     inv, att = ROPE_INIT_FUNCTIONS[rope_type](config, "cpu", seq_len=seq_len, **kwargs)
//!     record(name, reference or f"5.2 ROPE_INIT_FUNCTIONS[{rope_type!r}]", params, context, evaluated_context, inv, att, layer_type)
//!
//!
//! ctx = lambda head_dim=16, max_pos=128, **extra: dict(head_dim=head_dim, max_position_embeddings=max_pos, **extra)
//!
//! # Default and legacy-key precedence through a real Llama config.
//! p = {"rope_type": "default", "rope_theta": 10000.0}
//! c = llama(p)
//! from_embedding("llama_default", p, c, LlamaRotaryEmbedding(c), ctx(), 1, "5.2 LlamaRotaryEmbedding")
//!
//! p = {"type": "linear", "factor": 4.0}
//! c = llama(rope_scaling=copy.deepcopy(p), rope_theta=500000.0)
//! from_embedding("llama_legacy_linear_top_level_theta", p, c, LlamaRotaryEmbedding(c), ctx(rope_theta=500000.0), 1,
//!                "5.2 LlamaRotaryEmbedding (legacy rope_scaling/type and top-level rope_theta)")
//!
//! # Dynamic NTK: L is max_position_embeddings; frequencies change once the context exceeds it.
//! p = {"rope_type": "dynamic", "factor": 2.0, "rope_theta": 10000.0}
//! c = llama(p, max_position_embeddings=64)
//! from_init("llama_dynamic_within_context", p, c, ctx(max_pos=64), 64, seq_len=None)
//! c = llama(p, max_position_embeddings=64)
//! from_init("llama_dynamic_extended_context", p, c, ctx(max_pos=64), 150, seq_len=150)
//! p = {"rope_type": "dynamic", "factor": 3.0, "rope_theta": 50000.0, "partial_rotary_factor": 0.5}
//! c = llama(p, max_position_embeddings=64)
//! from_init("llama_dynamic_partial_rotary", p, c, ctx(max_pos=64), 200, seq_len=200)
//!
//! # YaRN variants.
//! p = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32, "rope_theta": 10000.0}
//! c = llama(p)
//! from_init("llama_yarn_defaults", p, c, ctx(), 1)
//! p = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32, "rope_theta": 10000.0,
//!      "beta_fast": 16.0, "beta_slow": 2.0, "attention_factor": 1.5}
//! c = llama(p)
//! from_init("llama_yarn_betas_and_attention_factor", p, c, ctx(), 1)
//! p = {"rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 32, "rope_theta": 10000.0,
//!      "mscale": 1.0, "mscale_all_dim": 0.8}
//! c = llama(p, max_position_embeddings=512)
//! from_init("llama_yarn_mscale_ratio", p, c, ctx(max_pos=512), 1)
//! p = {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 16, "rope_theta": 500000.0,
//!      "truncate": False, "partial_rotary_factor": 0.5}
//! c = llama(p)
//! from_init("llama_yarn_untruncated_partial_rotary", p, c, ctx(), 1)
//!
//! # DeepSeek-V2-Lite: Hugging Face rotary embedding, then the DeepSeek reference
//! # amplitude get_mscale(f, mscale or 1) / get_mscale(f, mscale_all_dim or 0)
//! # transcribed from modeling_deepseek.py (revision 604d5664dddd88a0433dbae533b7fe9472482de0),
//! # whose YaRN frequency formula is identical to Hugging Face's truncated YaRN.
//! fixture = json.load(open(f"{REPO}/tests/fixtures/deepseek_v2_lite.json"))
//! c = DeepseekV2Config.from_dict(copy.deepcopy(fixture))
//! e = DeepseekV2RotaryEmbedding(c)
//! deepseek_context = dict(head_dim=fixture["qk_rope_head_dim"], max_position_embeddings=fixture["max_position_embeddings"],
//!                         rope_theta=float(fixture["rope_theta"]))
//! from_embedding("deepseek_v2_lite_hugging_face", fixture["rope_scaling"], c, e, deepseek_context, 1,
//!                "5.2 DeepseekV2RotaryEmbedding on tests/fixtures/deepseek_v2_lite.json")
//!
//!
//! # Hugging Face reads mscale_all_dim alone only for the attention softmax scale;
//! # the rotary amplitude falls back to get_mscale(factor).
//! params = {k: v for k, v in fixture["rope_scaling"].items() if k != "mscale"}
//! c = DeepseekV2Config.from_dict({**copy.deepcopy(fixture), "rope_scaling": copy.deepcopy(params)})
//! from_embedding("deepseek_hugging_face_mscale_all_dim_only", params, c, DeepseekV2RotaryEmbedding(c), deepseek_context, 1,
//!                "5.2 DeepseekV2RotaryEmbedding without mscale")
//!
//!
//! def deepseek_mscale(scale, multiplier):
//!     return 1.0 if scale <= 1 else 0.1 * multiplier * math.log(scale) + 1.0
//!
//!
//! for name, overrides in [
//!     ("deepseek_v2_lite_reference", {}),
//!     ("deepseek_reference_mscale_only", {"mscale_all_dim": None}),
//!     ("deepseek_reference_mscale_all_dim_only", {"mscale": None}),
//! ]:
//!     params = dict(fixture["rope_scaling"])
//!     for key, value in overrides.items():
//!         params.pop(key)
//!     hf = DeepseekV2Config.from_dict({**copy.deepcopy(fixture), "rope_scaling": copy.deepcopy(params)})
//!     inv, _ = ROPE_INIT_FUNCTIONS["yarn"](hf, "cpu")
//!     amplitude = deepseek_mscale(params["factor"], params.get("mscale", 1)) / deepseek_mscale(params["factor"], params.get("mscale_all_dim", 0))
//!     record(name, "5.2 yarn frequencies with the DeepSeek reference mscale ratio", params,
//!            {**deepseek_context, "mscale_convention": "deep_seek_reference"}, 1, inv, amplitude)
//!
//! # LongRoPE: nested original/factor/attention, and Phi-3's top-level original.
//! short = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
//! long = [1.0, 1.5, 2.5, 4.0, 6.5, 9.0, 12.0, 16.0]
//! p = {"rope_type": "longrope", "short_factor": short, "long_factor": long, "original_max_position_embeddings": 32,
//!      "factor": 2.0, "attention_factor": 1.1, "rope_theta": 10000.0}
//! c = llama(p)
//! from_init("llama_longrope_short_table", p, c, ctx(), 32, seq_len=32)
//! from_init("llama_longrope_long_table", p, c, ctx(), 33, seq_len=33)
//! p = {"rope_type": "longrope", "short_factor": short, "long_factor": long}
//! phi = lambda: Phi3Config(hidden_size=64, num_attention_heads=4, max_position_embeddings=256,
//!                          original_max_position_embeddings=64, rope_parameters=copy.deepcopy(p))
//! c = phi()
//! phi_context = ctx(max_pos=256, original_max_position_embeddings=64)
//! from_init("phi3_longrope_top_level_original_short", p, c, phi_context, 64, seq_len=64)
//! from_init("phi3_longrope_top_level_original_long", p, c, phi_context, 65, seq_len=65)
//! c = Phi3Config(hidden_size=64, num_attention_heads=4, max_position_embeddings=256, partial_rotary_factor=0.5)
//! from_embedding("phi3_default_top_level_partial_rotary", None, c, Phi3RotaryEmbedding(c),
//!                ctx(max_pos=256, partial_rotary_factor=0.5), 1, "5.2 Phi3RotaryEmbedding")
//!
//! # Llama 3.1 band interpolation with wavelengths in all three bands.
//! p = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
//!      "original_max_position_embeddings": 32, "rope_theta": 500000.0}
//! c = llama(p, max_position_embeddings=256)
//! from_init("llama_llama3_bands", p, c, ctx(max_pos=256), 1)
//!
//! # Pinned checkpoint fixtures.
//! raw = json.load(open(f"{REPO}/tests/fixtures/qwen3_5_0_8b.json"))["text_config"]
//! c = Qwen3_5TextConfig(**copy.deepcopy(raw))
//! e = Qwen3_5TextRotaryEmbedding(c)
//! from_embedding("qwen3_5_0_8b_fixture", raw["rope_parameters"], c, e,
//!                dict(head_dim=raw["head_dim"], max_position_embeddings=raw["max_position_embeddings"], mrope="text_positions"),
//!                1, "5.2 Qwen3_5TextRotaryEmbedding on tests/fixtures/qwen3_5_0_8b.json")
//! raw = json.load(open(f"{REPO}/tests/fixtures/qwen3_30b_a3b.json"))
//! c = Qwen3MoeConfig.from_dict(copy.deepcopy(raw))
//! from_embedding("qwen3_30b_a3b_fixture", raw["rope_scaling"], c, Qwen3MoeRotaryEmbedding(c),
//!                dict(head_dim=raw["head_dim"], max_position_embeddings=raw["max_position_embeddings"],
//!                     rope_theta=float(raw["rope_theta"])), 1, "5.2 Qwen3MoeRotaryEmbedding on tests/fixtures/qwen3_30b_a3b.json")
//!
//! # Tiny end-to-end goldens: config.json through AutoConfig and the family's rotary module.
//! def tiny(directory):
//!     path = f"{REPO}/tests/fixtures/tiny_goldens/{directory}"
//!     return AutoConfig.from_pretrained(path), json.load(open(f"{path}/config.json"))
//!
//!
//! c, raw = tiny("llama3")
//! from_embedding("tiny_golden_llama3", raw["rope_parameters"], c, LlamaRotaryEmbedding(c),
//!                ctx(raw["head_dim"], raw["max_position_embeddings"]), 1, "5.2 LlamaRotaryEmbedding on tiny_goldens/llama3")
//! c, raw = tiny("gemma3-text")
//! e = Gemma3RotaryEmbedding(c)
//! for layer_type in ["full_attention", "sliding_attention"]:
//!     record(f"tiny_golden_gemma3_text_{layer_type}", "5.2 Gemma3RotaryEmbedding on tiny_goldens/gemma3-text",
//!            raw["rope_parameters"], ctx(raw["head_dim"], raw["max_position_embeddings"]), 1,
//!            getattr(e, f"{layer_type}_inv_freq"), getattr(e, f"{layer_type}_attention_scaling"), layer_type)
//! c, raw = tiny("gpt-oss")
//! from_embedding("tiny_golden_gpt_oss", raw["rope_parameters"], c, GptOssRotaryEmbedding(c),
//!                ctx(raw["head_dim"], raw["max_position_embeddings"]), 1, "5.2 GptOssRotaryEmbedding on tiny_goldens/gpt-oss")
//!
//! # Proportional RoPE (Transformers 5.8.0), Gemma-4-style per-layer mapping.
//! p = {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
//!      "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}}
//! for name, params in [("proportional_quarter", p),
//!                      ("proportional_scaled", {"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.6,
//!                                                                  "factor": 2.0, "rope_theta": 10000.0}})]:
//!     c = PretrainedConfig()
//!     c.hidden_size, c.num_attention_heads, c.head_dim, c.max_position_embeddings = 64, 4, 16, 128
//!     c.layer_types = list(params)
//!     c.rope_parameters = copy.deepcopy(params)
//!     inv, att = rope58._compute_proportional_rope_parameters(c, "cpu", layer_type="full_attention")
//!     record(name, "5.8.0 _compute_proportional_rope_parameters", params, ctx(), 1, inv, att, "full_attention")
//!
//! json.dump({"transformers": transformers.__version__, "torch": torch.__version__, "cases": cases}, sys.stdout, indent=1)
//! print()
//! ```

use serde::Deserialize;

use super::{MropeHandling, RopeContext, RopeParameterSet, RopeType, YarnMscaleConvention};
use crate::position::RotaryFrequencies;

/// Largest accepted relative inverse-frequency error against the F32 reference.
const FREQUENCY_TOLERANCE: f64 = 1e-6;
/// Largest accepted relative amplitude error; both sides compute in F64.
const AMPLITUDE_TOLERANCE: f64 = 1e-12;

/// Generated reference data with the library versions that produced it.
#[derive(Deserialize)]
struct GoldenFile {
    /// Transformers version of the main reference.
    transformers: String,
    /// Torch version used by the reference.
    torch: String,
    /// One translated configuration per entry.
    cases: Vec<GoldenCase>,
}

/// One reference configuration and its expected tables.
#[derive(Deserialize)]
struct GoldenCase {
    /// Stable case name used in assertion messages.
    name: String,
    /// Reference class or function that produced the expectation.
    reference: String,
    /// Raw serialized `rope_parameters`/`rope_scaling`, or `null` when absent.
    rope_parameters: Option<RopeParameterSet>,
    /// Layer type for per-layer-type mappings.
    layer_type: Option<String>,
    /// Surrounding config fields passed to the translator.
    context: GoldenContext,
    /// Evaluated context (largest position plus one) of the reference tables.
    evaluated_context: usize,
    /// Reference inverse frequencies, one per rotated pair.
    inv_freq: Vec<f64>,
    /// Reference cosine/sine amplitude.
    attention_scaling: f64,
}

/// Surrounding config fields and conventions of one golden case.
#[derive(Deserialize)]
struct GoldenContext {
    /// Full head width.
    head_dim: usize,
    /// Serialized maximum context.
    max_position_embeddings: usize,
    /// Legacy top-level theta.
    #[serde(default)]
    rope_theta: Option<f64>,
    /// Legacy top-level partial rotary factor.
    #[serde(default)]
    partial_rotary_factor: Option<f64>,
    /// Phi-3-style top-level original context.
    #[serde(default)]
    original_max_position_embeddings: Option<usize>,
    /// `deep_seek_reference` selects the DeepSeek amplitude convention.
    #[serde(default)]
    mscale_convention: Option<String>,
    /// `text_positions` accepts multimodal section metadata.
    #[serde(default)]
    mrope: Option<String>,
}

impl GoldenContext {
    /// Builds the translator context this case describes.
    fn rope_context(&self) -> RopeContext {
        let convention = match self.mscale_convention.as_deref() {
            None => YarnMscaleConvention::HuggingFace,
            Some("deep_seek_reference") => YarnMscaleConvention::DeepSeekReference,
            Some(other) => panic!("unknown mscale convention {other}"),
        };
        let mrope = match self.mrope.as_deref() {
            None => MropeHandling::Reject,
            Some("text_positions") => MropeHandling::TextPositions,
            Some(other) => panic!("unknown mrope handling {other}"),
        };
        RopeContext::new("golden", self.head_dim, self.max_position_embeddings)
            .with_rope_theta(self.rope_theta)
            .with_partial_rotary_factor(self.partial_rotary_factor)
            .with_original_max_position_embeddings(self.original_max_position_embeddings)
            .with_yarn_mscale_convention(convention)
            .with_mrope(mrope)
    }
}

/// Parses the checked-in golden file.
fn goldens() -> GoldenFile {
    serde_json::from_str(include_str!("rope_goldens.json")).expect("valid golden JSON")
}

/// Returns the raw `rope_parameters` of the repository fixture a case was generated from.
fn fixture_rope_parameters(name: &str) -> Option<serde_json::Value> {
    let (source, pointer) = match name {
        "tiny_golden_llama3" => (
            include_str!("../../../tests/fixtures/tiny_goldens/llama3/config.json"),
            "/rope_parameters",
        ),
        "tiny_golden_gemma3_text_full_attention" | "tiny_golden_gemma3_text_sliding_attention" => (
            include_str!("../../../tests/fixtures/tiny_goldens/gemma3-text/config.json"),
            "/rope_parameters",
        ),
        "tiny_golden_gpt_oss" => (
            include_str!("../../../tests/fixtures/tiny_goldens/gpt-oss/config.json"),
            "/rope_parameters",
        ),
        "deepseek_v2_lite_hugging_face" | "deepseek_v2_lite_reference" => (
            include_str!("../../../tests/fixtures/deepseek_v2_lite.json"),
            "/rope_scaling",
        ),
        "qwen3_5_0_8b_fixture" => (
            include_str!("../../../tests/fixtures/qwen3_5_0_8b.json"),
            "/text_config/rope_parameters",
        ),
        "qwen3_30b_a3b_fixture" => (
            include_str!("../../../tests/fixtures/qwen3_30b_a3b.json"),
            "/rope_scaling",
        ),
        _ => return None,
    };
    let value: serde_json::Value = serde_json::from_str(source).expect("fixture JSON");
    Some(value.pointer(pointer).expect("fixture RoPE field").clone())
}

/// Every golden case translates to the reference frequencies and amplitude.
#[test]
fn translated_configs_match_transformers_frequencies_and_amplitude() {
    let file = goldens();
    assert_eq!(file.transformers, "5.2.0");
    assert_eq!(file.torch, "2.10.0");
    let mut worst = 0.0_f64;
    for case in &file.cases {
        let context = case.context.rope_context();
        let set = case.rope_parameters.clone().unwrap_or_default();
        let layer_type = case.layer_type.as_deref().unwrap_or("full_attention");
        let config = set
            .to_rope_config(layer_type, &context)
            .unwrap_or_else(|error| panic!("{} ({}): {error}", case.name, case.reference));
        let frequencies = RotaryFrequencies::new(config, case.evaluated_context)
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        assert_eq!(
            frequencies.inverse().len(),
            case.inv_freq.len(),
            "{}: rotated pair count",
            case.name
        );
        for (index, (&actual, &expected)) in
            frequencies.inverse().iter().zip(&case.inv_freq).enumerate()
        {
            let actual = f64::from(actual);
            if expected == 0.0 {
                assert_eq!(actual, 0.0, "{} pair {index}", case.name);
                continue;
            }
            let relative = ((actual - expected) / expected).abs();
            worst = worst.max(relative);
            assert!(
                relative <= FREQUENCY_TOLERANCE,
                "{} pair {index}: {actual} vs {expected} (relative {relative:e})",
                case.name
            );
        }
        let amplitude_error =
            ((frequencies.amplitude() - case.attention_scaling) / case.attention_scaling).abs();
        assert!(
            amplitude_error <= AMPLITUDE_TOLERANCE,
            "{} amplitude: {} vs {}",
            case.name,
            frequencies.amplitude(),
            case.attention_scaling
        );
    }
    assert!(
        worst <= FREQUENCY_TOLERANCE,
        "worst relative error {worst:e}"
    );
}

/// Fixture-derived cases carry exactly the metadata checked into `tests/fixtures`.
#[test]
fn fixture_cases_use_the_checked_in_fixture_metadata() {
    let file = goldens();
    let mut checked = 0;
    for case in &file.cases {
        let Some(expected) = fixture_rope_parameters(&case.name) else {
            continue;
        };
        let expected: Option<RopeParameterSet> =
            serde_json::from_value(expected).expect("fixture RoPE metadata parses");
        assert_eq!(case.rope_parameters, expected, "{}", case.name);
        checked += 1;
    }
    assert_eq!(checked, 8, "every fixture-derived case was compared");
}

/// The golden set exercises every algorithm and both mapping shapes.
#[test]
fn goldens_cover_every_rope_type_and_mapping_shape() {
    let file = goldens();
    let mut seen = std::collections::BTreeSet::new();
    let mut per_layer = 0;
    for case in &file.cases {
        let set = case.rope_parameters.clone().unwrap_or_default();
        if matches!(set, RopeParameterSet::ByLayerType(_)) {
            per_layer += 1;
        }
        let layer_type = case.layer_type.as_deref().unwrap_or("full_attention");
        let parameters = set.parameters_for(layer_type).expect("layer parameters");
        seen.insert(parameters.rope_type.as_str());
    }
    for rope_type in [
        RopeType::Default,
        RopeType::Linear,
        RopeType::Dynamic,
        RopeType::Yarn,
        RopeType::LongRope,
        RopeType::Llama3,
        RopeType::Proportional,
    ] {
        assert!(
            seen.contains(rope_type.as_str()),
            "{rope_type:?} has no golden"
        );
    }
    assert!(per_layer >= 4, "per-layer-type cases: {per_layer}");
}
