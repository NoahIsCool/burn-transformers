//! Bounded on-disk shard/index parsing; file names written by this harness are fixed.

use std::fs;
use transformers::checkpoint::{
    ArtifactLimits, ArtifactSelection, CheckpointManifest, LoadOptions,
};

/// Wraps arbitrary header JSON in a physically tiny SafeTensors envelope.
fn tensor_file(header: &[u8]) -> Vec<u8> {
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header);
    file.extend_from_slice(&1.0_f32.to_le_bytes());
    file
}

/// Fixed valid scalar shard used to explore the index parser beyond malformed-JSON rejection.
fn valid_shard() -> Vec<u8> {
    tensor_file(br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#)
}

/// Discovers one disposable two-file artifact and checks successful inventory invariants.
///
/// Input mode selects raw index JSON, a shard-path string, a raw shard envelope, or raw header
/// JSON. Only `a.safetensors` and the standard index basename are ever written. Discovery must
/// reject any untrusted reference that escapes the temporary checkpoint directory.
/// Returns whether discovery succeeded, allowing deterministic tests to prove both branches run.
pub fn exercise(input: &[u8]) -> bool {
    let input = &input[..input.len().min(crate::MAX_INPUT_BYTES)];
    let (&mode, payload) = input.split_first().unwrap_or((&0, &[]));
    let index = br#"{"metadata":{"total_size":4},"weight_map":{"weight":"a.safetensors"}}"#;
    let (index, shard) = match mode % 4 {
        0 => (payload.to_vec(), valid_shard()),
        1 => (
            serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": 4},
                "weight_map": {"weight": String::from_utf8_lossy(payload)}
            }))
            .unwrap(),
            valid_shard(),
        ),
        2 => (index.to_vec(), payload.to_vec()),
        _ => (index.to_vec(), tensor_file(payload)),
    };
    let directory = tempfile::tempdir().expect("create isolated artifact fuzz directory");
    fs::write(directory.path().join("a.safetensors"), shard).unwrap();
    fs::write(directory.path().join("model.safetensors.index.json"), index).unwrap();
    let options = LoadOptions {
        artifact: ArtifactSelection::Index("model.safetensors.index.json".into()),
        limits: ArtifactLimits {
            max_header_bytes: 4096,
            max_index_bytes: 8192,
            max_shards: 4,
            max_tensors: 64,
            max_rank: 8,
            max_metadata_entries: 16,
        },
        ..LoadOptions::default()
    };
    if let Ok(manifest) = CheckpointManifest::discover(directory.path(), &options) {
        let root = directory.path().canonicalize().unwrap();
        assert_eq!(manifest.root, root);
        assert!(manifest.shards.len() <= 4);
        assert!(manifest.tensors.len() <= 64);
        let mut total = 0_u64;
        for shard in &manifest.shards {
            assert!(shard.path.starts_with(&root));
            assert_eq!(shard.path.file_name().unwrap(), "a.safetensors");
            assert!(shard.header_bytes <= 4096);
            for tensor in shard.tensors.values() {
                assert!(tensor.data_offsets[0] <= tensor.data_offsets[1]);
                assert!(tensor.data_offsets[1] <= shard.data_bytes);
                assert_eq!(
                    tensor.byte_len,
                    tensor.data_offsets[1] - tensor.data_offsets[0]
                );
                total = total.checked_add(tensor.byte_len).unwrap();
            }
        }
        assert_eq!(total, manifest.tensor_bytes);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seeds exercise valid discovery, traversal, duplicate keys and corrupt envelopes.
    #[test]
    fn artifact_modes_have_bounded_success_and_failure_paths() {
        assert!(exercise(b"\x01a.safetensors"));
        assert!(!exercise(b"\x01../a.safetensors"));
        assert!(!exercise(b"\x01/etc/passwd"));
        assert!(!exercise(b"\0{\"weight_map\":{},\"weight_map\":{}}"));
        assert!(!exercise(b"\x02\xff\xff\xff\xff\xff\xff\xff\xff"));
        assert!(exercise(
            b"\x03{\"weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}"
        ));
    }
}
