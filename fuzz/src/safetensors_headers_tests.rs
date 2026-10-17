//! Deterministic regression seeds for the SafeTensors header and index fuzz harness.

use super::*;

/// A shard whose tensors are given and whose offsets are contiguous in order.
fn contiguous(tensors: Vec<TensorSpec>) -> ShardSpec {
    let mut tensors = tensors;
    let covered = assign_offsets(&mut Bytes::new(&[]), &mut tensors);
    ShardSpec {
        tensors,
        metadata: None,
        metadata_position: 0,
        padding: 0,
        data_len: covered,
        prefix: Prefix::Exact,
    }
}

/// A tensor spec with canonical integer dimensions.
fn spec(name: &str, dtype: &'static str, shape: &[u64]) -> TensorSpec {
    let width = DTYPES
        .iter()
        .find(|(spelling, _)| *spelling == dtype)
        .map_or(0, |(_, bits)| *bits);
    TensorSpec {
        name: name.to_owned(),
        dtype: (dtype, width),
        shape: shape.iter().copied().map(Lexeme::integer).collect(),
        offsets: [Lexeme::integer(0), Lexeme::integer(0)],
    }
}

/// Generous limits that never bind.
fn generous() -> ArtifactLimits {
    ArtifactLimits {
        max_header_bytes: 4096,
        max_index_bytes: 8192,
        max_shards: 4,
        max_tensors: 64,
        max_rank: 8,
        max_metadata_entries: 16,
    }
}

/// Discovers `spec` as `model.safetensors` and returns whether it was accepted.
fn discover(spec: &ShardSpec, limits: &ArtifactLimits) -> bool {
    let directory = tempfile::tempdir().unwrap();
    fs::write(directory.path().join("model.safetensors"), spec.file()).unwrap();
    let options = LoadOptions {
        artifact: ArtifactSelection::Single("model.safetensors".into()),
        limits: limits.clone(),
        ..LoadOptions::default()
    };
    let accepted = CheckpointManifest::discover(directory.path(), &options).is_ok();
    assert_eq!(
        accepted,
        expected_shard(spec, limits).is_some(),
        "{}",
        spec.header()
    );
    accepted
}

/// MXFP4 layouts: packed `U8` blocks, `F4` values with `F8_E8M0` scales, odd `F4` counts.
#[test]
fn microscaling_layouts_follow_exact_byte_accounting() {
    let packed = contiguous(vec![
        spec("blocks", "U8", &[2, 3, 2, 16]),
        spec("scales", "U8", &[2, 3, 2]),
    ]);
    assert!(discover(&packed, &generous()));
    let microscaling = contiguous(vec![
        spec("blocks", "F4", &[2, 3, 64]),
        spec("scales", "F8_E8M0", &[2, 3, 2]),
        spec("six", "F6_E2M3", &[4]),
    ]);
    assert!(discover(&microscaling, &generous()));
    // 63 four-bit values end in half a byte.
    assert!(!discover(
        &contiguous(vec![spec("odd", "F4", &[63])]),
        &generous()
    ));
    // Six-bit values need a multiple of four elements.
    assert!(!discover(
        &contiguous(vec![spec("six", "F6_E3M2", &[2])]),
        &generous()
    ));
}

/// Overflowing products, unknown dtypes, and bad lexemes are rejected.
#[test]
fn malformed_entries_are_rejected() {
    for tensor in [
        spec("overflow", "U8", &[1 << 62, 8, 0]),
        spec("bits", "U64", &[1 << 61]),
        spec("unknown", "F8_E4M3FN", &[1]),
        spec("", "U8", &[1]),
    ] {
        assert!(!discover(&contiguous(vec![tensor]), &generous()));
    }
    let mut lexeme = contiguous(vec![spec("w", "U8", &[4])]);
    lexeme.tensors[0].shape[0] = Lexeme {
        text: "4.0".into(),
        value: None,
    };
    assert!(!discover(&lexeme, &generous()));
}

/// Gaps, overlaps, trailing bytes, and framing errors are rejected; padding is not.
#[test]
fn tiling_and_framing_are_exact() {
    let base = || contiguous(vec![spec("a", "U8", &[3]), spec("b", "F32", &[2])]);
    assert!(discover(&base(), &generous()));
    let mut trailing = base();
    trailing.data_len += 1;
    assert!(!discover(&trailing, &generous()));
    let mut gap = base();
    gap.tensors[1].offsets = [Lexeme::integer(4), Lexeme::integer(12)];
    gap.data_len = 12;
    assert!(!discover(&gap, &generous()));
    let mut padded = base();
    padded.padding = 3;
    assert!(discover(&padded, &generous()));
    // One byte shorter drops a padding space into the data section.
    padded.prefix = Prefix::Shorter;
    assert!(!discover(&padded, &generous()));
    for prefix in [Prefix::Longer, Prefix::Zero, Prefix::Maximum] {
        let mut framed = base();
        framed.prefix = prefix;
        assert!(!discover(&framed, &generous()));
    }
}

/// Metadata must be a unique string map within its limit.
#[test]
fn metadata_entries_are_strings_and_bounded() {
    let mut metadata = contiguous(vec![spec("w", "U8", &[1])]);
    metadata.metadata = Some(vec![("format".into(), "\"pt\"".into())]);
    assert!(discover(&metadata, &generous()));
    metadata.metadata_position = 1;
    assert!(discover(&metadata, &generous()));
    let tight = ArtifactLimits {
        max_metadata_entries: 0,
        ..generous()
    };
    assert!(!discover(&metadata, &tight));
    for entries in [
        vec![("format".to_owned(), "1".to_owned())],
        vec![("format".to_owned(), "\"pt\"".to_owned()); 2],
    ] {
        metadata.metadata = Some(entries);
        assert!(!discover(&metadata, &generous()));
    }
}

/// Committed seeds keep their meaning: each is a valid artifact.
#[test]
fn committed_seeds_pass() {
    for seed in [
        &include_bytes!("../seeds/safetensors_headers/mxfp4-pair")[..],
        &include_bytes!("../seeds/safetensors_headers/metadata-padding")[..],
        &include_bytes!("../seeds/safetensors_headers/index-alias")[..],
    ] {
        assert!(exercise(seed), "{seed:?}");
    }
    exercise(&[]);
    exercise(&[255; 128]);
}

/// Pseudo-random inputs cover single files and indexes, accepted and rejected.
#[test]
fn pseudo_random_inputs_agree_with_the_oracle() {
    let inputs = crate::lcg_inputs(0x7361_6665, 768, 64);
    let mut outcomes = BTreeMap::new();
    for input in &inputs {
        let accepted = exercise(input);
        *outcomes.entry((input[0] >= 160, accepted)).or_insert(0) += 1;
    }
    // Every (mode, outcome) pair occurs, so neither oracle branch is vacuous.
    assert_eq!(outcomes.len(), 4, "{outcomes:?}");
}
