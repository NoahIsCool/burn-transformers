//! Parser and topology oracles that never construct model parameter tensors.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use transformers::{
    architecture::{DecoderSpec, Seq2SeqSpec},
    models::{
        config::{ConfigError, ModelConfig, Validate, parse_text_config},
        deepseek_v2::DeepseekV2Config,
        muse_glimmer::MuseGlimmerTextConfig,
        nemotron_h::NemotronHConfig,
        qwen3_5::Qwen3_5TextConfig,
        qwen3_moe::Qwen3MoeConfig,
        t5::T5Config,
    },
};

/// JSON value wrapper that exercises the legacy Infinity lexer without topology loops.
#[derive(Deserialize)]
#[serde(transparent)]
struct LexicalValue(Value);

impl Validate for LexicalValue {
    /// Arbitrary parsed JSON has no model-shape invariants.
    fn validate(&self) -> Result<(), ConfigError> {
        let _ = &self.0;
        Ok(())
    }
}

/// Bounds only counters that can control loops; extreme matrix dimensions remain adversarial.
fn bounded_work(value: &Value) -> bool {
    match value {
        Value::Object(fields) => fields.iter().all(|(name, value)| {
            let layer_count = matches!(
                name.as_str(),
                "num_hidden_layers"
                    | "num_layers"
                    | "num_decoder_layers"
                    | "num_nextn_predict_layers"
            );
            (!layer_count || value.as_u64().is_none_or(|count| count <= 128)) && bounded_work(value)
        }),
        Value::Array(values) => values.iter().all(bounded_work),
        _ => true,
    }
}

/// Successful family parsing must survive serialization and preserve its normalized topology.
fn roundtrip<T: ModelConfig + Serialize>(json: &str) {
    if let Ok(config) = T::from_json_str(json) {
        let encoded = serde_json::to_string(&config).expect("valid family serialization");
        let decoded = T::from_json_str(&encoded).expect("serialized family still validates");
        assert_eq!(
            serde_json::to_value(config).unwrap(),
            serde_json::to_value(decoded).unwrap()
        );
    }
}

/// Exercises all existing family parsers and modular topology validators without allocation.
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
    if !bounded_work(&value) {
        return;
    }
    roundtrip::<Qwen3_5TextConfig>(json);
    roundtrip::<Qwen3MoeConfig>(json);
    roundtrip::<DeepseekV2Config>(json);
    roundtrip::<MuseGlimmerTextConfig>(json);
    roundtrip::<NemotronHConfig>(json);
    roundtrip::<T5Config>(json);
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
