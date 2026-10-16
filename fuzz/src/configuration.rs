//! Parser and topology oracles that never construct model parameter tensors.
//!
//! Every registered family parser reads the same input: the Llama family (six
//! `model_type`s), Gemma 2 and 3, gpt-oss, GPT-2, GPT-NeoX, Qwen3.5, Qwen3-MoE,
//! DeepSeek-V2, Muse Glimmer, Nemotron-H, and T5. An accepted configuration
//! must survive its normalized serialization round trip, and every layer and
//! routed-expert count it accepted must lie within the shared bounds
//! [`MAX_LAYERS`] and [`MAX_EXPERTS`]. Inputs are not filtered by their
//! counts: absurd values (the class that once overflowed vector capacities
//! during validation) must be rejected with a contextual error, quickly and
//! without panicking, while counts within the bounds still exercise the
//! per-layer schedules they size.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use transformers::{
    architecture::{DecoderSpec, Seq2SeqSpec},
    models::{
        config::{ConfigError, MAX_EXPERTS, MAX_LAYERS, ModelConfig, Validate, parse_text_config},
        deepseek_v2::DeepseekV2Config,
        gemma::GemmaConfig,
        gpt_neox::GptNeoXConfig,
        gpt_oss::GptOssConfig,
        gpt2::Gpt2Config,
        llama::LlamaFamilyConfig,
        muse_glimmer::MuseGlimmerTextConfig,
        nemotron_h::{NemotronHBlockType, NemotronHConfig},
        qwen3_5::Qwen3_5TextConfig,
        qwen3_moe::Qwen3MoeConfig,
        t5::T5Config,
    },
};

/// JSON value wrapper that exercises the legacy Infinity lexer without topology loops.
#[derive(Deserialize)]
#[serde(transparent)]
struct LexicalValue(
    /// Parsed JSON payload, retained only to validate lexer acceptance.
    Value,
);

impl Validate for LexicalValue {
    /// Arbitrary parsed JSON has no model-shape invariants.
    fn validate(&self) -> Result<(), ConfigError> {
        let _ = &self.0;
        Ok(())
    }
}

/// One count an accepted configuration sizes structures by, with its shared bound.
struct BoundedCount {
    /// Serialized field name, for assertion messages.
    field: &'static str,
    /// Accepted value.
    value: usize,
    /// Shared bound the value must not exceed.
    bound: usize,
}

impl BoundedCount {
    /// A layer count, bounded by [`MAX_LAYERS`].
    const fn layers(field: &'static str, value: usize) -> Self {
        Self {
            field,
            value,
            bound: MAX_LAYERS,
        }
    }

    /// A routed-expert count, bounded by [`MAX_EXPERTS`].
    const fn experts(field: &'static str, value: usize) -> Self {
        Self {
            field,
            value,
            bound: MAX_EXPERTS,
        }
    }
}

/// The counts of an accepted family configuration that size per-layer or per-expert work.
trait BoundedCounts {
    /// Lists every layer and routed-expert count the family reads.
    fn bounded_counts(&self) -> Vec<BoundedCount>;
}

impl BoundedCounts for LlamaFamilyConfig {
    /// One decoder stack.
    fn bounded_counts(&self) -> Vec<BoundedCount> {
        vec![BoundedCount::layers(
            "num_hidden_layers",
            self.num_hidden_layers,
        )]
    }
}

impl BoundedCounts for GemmaConfig {
    /// One decoder stack.
    fn bounded_counts(&self) -> Vec<BoundedCount> {
        vec![BoundedCount::layers(
            "num_hidden_layers",
            self.num_hidden_layers,
        )]
    }
}

impl BoundedCounts for GptOssConfig {
    /// One decoder stack with fused per-layer expert banks.
    fn bounded_counts(&self) -> Vec<BoundedCount> {
        vec![
            BoundedCount::layers("num_hidden_layers", self.num_hidden_layers),
            BoundedCount::experts("num_local_experts", self.num_local_experts),
        ]
    }
}

impl BoundedCounts for Gpt2Config {
    /// One decoder stack.
    fn bounded_counts(&self) -> Vec<BoundedCount> {
        vec![BoundedCount::layers("n_layer", self.n_layer)]
    }
}

impl BoundedCounts for GptNeoXConfig {
    /// One decoder stack.
    fn bounded_counts(&self) -> Vec<BoundedCount> {
        vec![BoundedCount::layers(
            "num_hidden_layers",
            self.num_hidden_layers,
        )]
    }
}

impl BoundedCounts for Qwen3_5TextConfig {
    /// The decoder stack and its multi-token-prediction blocks.
    fn bounded_counts(&self) -> Vec<BoundedCount> {
        vec![
            BoundedCount::layers("num_hidden_layers", self.num_hidden_layers),
            BoundedCount::layers("mtp_num_hidden_layers", self.mtp_num_hidden_layers),
        ]
    }
}

impl BoundedCounts for Qwen3MoeConfig {
    /// One decoder stack with per-expert banks.
    fn bounded_counts(&self) -> Vec<BoundedCount> {
        vec![
            BoundedCount::layers("num_hidden_layers", self.num_hidden_layers),
            BoundedCount::experts("num_experts", self.num_experts),
        ]
    }
}

impl BoundedCounts for DeepseekV2Config {
    /// One decoder stack with per-expert banks.
    fn bounded_counts(&self) -> Vec<BoundedCount> {
        vec![
            BoundedCount::layers("num_hidden_layers", self.num_hidden_layers),
            BoundedCount::experts("n_routed_experts", self.n_routed_experts),
        ]
    }
}

impl BoundedCounts for MuseGlimmerTextConfig {
    /// One decoder stack.
    fn bounded_counts(&self) -> Vec<BoundedCount> {
        vec![BoundedCount::layers(
            "num_hidden_layers",
            self.num_hidden_layers,
        )]
    }
}

impl BoundedCounts for NemotronHConfig {
    /// The block schedule, and the expert bank whenever an MoE block reads it.
    fn bounded_counts(&self) -> Vec<BoundedCount> {
        let mut counts = vec![BoundedCount::layers(
            "layers_block_type",
            self.block_count(),
        )];
        let moe = self.layers_block_type.contains(&NemotronHBlockType::Moe)
            || (self.num_nextn_predict_layers > 0
                && self
                    .mtp_layers_block_type
                    .contains(&NemotronHBlockType::Moe));
        if moe {
            counts.push(BoundedCount::experts(
                "n_routed_experts",
                self.n_routed_experts,
            ));
        }
        counts
    }
}

impl BoundedCounts for T5Config {
    /// The encoder and decoder stacks.
    fn bounded_counts(&self) -> Vec<BoundedCount> {
        vec![
            BoundedCount::layers("num_layers", self.num_layers),
            BoundedCount::layers("num_decoder_layers", self.decoder_layers()),
        ]
    }
}

/// Parses `json` as `T`; an accepted configuration must respect the count bounds and round-trip.
///
/// Returns whether `T` accepted the document. Rejection is always allowed;
/// a panic, an accepted count beyond its bound, or a round trip that fails
/// or changes the normalized topology is a finding.
fn roundtrip<T: ModelConfig + Serialize + BoundedCounts>(json: &str) -> bool {
    let Ok(config) = T::from_json_str(json) else {
        return false;
    };
    for count in config.bounded_counts() {
        assert!(
            count.value <= count.bound,
            "{} accepted {} = {} above the shared bound {}",
            T::FAMILY,
            count.field,
            count.value,
            count.bound
        );
    }
    let encoded = serde_json::to_string(&config).expect("valid family serialization");
    let decoded = T::from_json_str(&encoded).expect("serialized family still validates");
    assert_eq!(
        serde_json::to_value(config).unwrap(),
        serde_json::to_value(decoded).unwrap()
    );
    true
}

/// A family's diagnostic name and its checked round trip, which reports acceptance.
type FamilyParser = (&'static str, fn(&str) -> bool);

/// Runs every registered family parser on `json` and returns the families that accepted it.
fn every_family(json: &str) -> Vec<&'static str> {
    /// Each family's diagnostic name and its checked round trip.
    const PARSERS: [FamilyParser; 11] = [
        (LlamaFamilyConfig::FAMILY, roundtrip::<LlamaFamilyConfig>),
        (GemmaConfig::FAMILY, roundtrip::<GemmaConfig>),
        (GptOssConfig::FAMILY, roundtrip::<GptOssConfig>),
        (Gpt2Config::FAMILY, roundtrip::<Gpt2Config>),
        (GptNeoXConfig::FAMILY, roundtrip::<GptNeoXConfig>),
        (Qwen3_5TextConfig::FAMILY, roundtrip::<Qwen3_5TextConfig>),
        (Qwen3MoeConfig::FAMILY, roundtrip::<Qwen3MoeConfig>),
        (DeepseekV2Config::FAMILY, roundtrip::<DeepseekV2Config>),
        (
            MuseGlimmerTextConfig::FAMILY,
            roundtrip::<MuseGlimmerTextConfig>,
        ),
        (NemotronHConfig::FAMILY, roundtrip::<NemotronHConfig>),
        (T5Config::FAMILY, roundtrip::<T5Config>),
    ];
    PARSERS
        .into_iter()
        .filter_map(|(family, parse)| parse(json).then_some(family))
        .collect()
}

/// Exercises every family parser and the modular topology validators without allocation.
pub fn exercise(input: &[u8]) {
    if input.len() > crate::MAX_INPUT_BYTES {
        return;
    }
    let Ok(json) = core::str::from_utf8(input) else {
        return;
    };
    let parsed = parse_text_config::<LexicalValue>(json);
    let Ok(LexicalValue(value)) = parsed else {
        return;
    };
    every_family(json);
    if let Ok(config) = serde_json::from_value::<DecoderSpec>(value.clone()) {
        let _ = config.validate();
    }
    if let Ok(config) = serde_json::from_value::<Seq2SeqSpec>(value) {
        let _ = config.validate();
    }
}

#[cfg(test)]
#[path = "configuration_tests.rs"]
mod tests;
