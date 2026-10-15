//! SafeTensors headers and shard indexes against an independent byte-accounting oracle.
//!
//! [`CheckpointManifest::discover`] inventories local SafeTensors artifacts
//! without reading payloads. Unlike `index_artifacts`, which mutates raw JSON,
//! this harness *generates* well-formed and subtly malformed artifacts and
//! predicts the exact outcome:
//!
//! - **Single-file mode** writes `model.safetensors` with up to four tensors of
//!   any SafeTensors dtype (including the microscaling `F4`, `F6_*`, and
//!   `F8_E8M0` formats and MXFP4-style `*_blocks`/`*_scales` pairs), shapes up
//!   to rank four with overflow-boundary dimensions, contiguous or perturbed
//!   offsets, optional `__metadata__`, duplicate keys, header padding, and a
//!   perturbed length prefix or data section. [`expected_shard`] restates the
//!   documented rules (exact `product(shape) * bits / 8` byte lengths with
//!   overflow and partial-byte rejection, contiguous tiling of the whole data
//!   section, unique names, limits) and must agree with discovery, including
//!   every inventoried field. Whenever discovery accepts a file, the official
//!   `safetensors` 0.7 parser must accept it too and report the same tensors.
//! - **Index mode** writes up to three valid shards plus
//!   `model.safetensors.index.json` with a perturbed `weight_map` (dropped,
//!   phantom, moved, duplicated, or unsafe entries; authorized and unauthorized
//!   index-only aliases) and an exact, wrong, or oddly spelled
//!   `metadata.total_size`. [`expected_index`] predicts acceptance and the
//!   manifest totals under logical and physical alias accounting.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};

use transformers::checkpoint::{
    ArtifactLimits, ArtifactSelection, CheckpointManifest, IndexAliasSizeAccounting,
    IndexTensorAlias, LoadOptions, TensorDType,
};

use crate::Bytes;

/// Every SafeTensors dtype spelling with its bit width, plus rejected spellings (width 0).
const DTYPES: [(&str, u64); 24] = [
    ("BOOL", 8),
    ("F4", 4),
    ("F6_E2M3", 6),
    ("F6_E3M2", 6),
    ("U8", 8),
    ("I8", 8),
    ("F8_E5M2", 8),
    ("F8_E4M3", 8),
    ("F8_E8M0", 8),
    ("I16", 16),
    ("U16", 16),
    ("F16", 16),
    ("BF16", 16),
    ("I32", 32),
    ("U32", 32),
    ("F32", 32),
    ("C64", 64),
    ("F64", 64),
    ("I64", 64),
    ("U64", 64),
    ("F8_E4M3FN", 0),
    ("MXFP4", 0),
    ("f32", 0),
    ("", 0),
];
/// Dimension values: small sizes, MXFP4 block widths, and `u64` overflow boundaries.
const DIMENSIONS: [u64; 12] = [
    0,
    1,
    2,
    3,
    4,
    16,
    32,
    90,
    1 << 31,
    (1 << 32) + 1,
    1 << 62,
    u64::MAX,
];
/// Tensor names, including an empty name and a name shared by two tensors.
const NAMES: [&str; 6] = [
    "w",
    "model.layers.0.mlp.experts.gate_up_proj_blocks",
    "model.layers.0.mlp.experts.gate_up_proj_scales",
    "é.weight",
    "",
    "w",
];
/// Largest data section the harness writes.
const MAX_DATA: u64 = 1 << 16;
/// Payload byte; never JSON whitespace, so a too-long header prefix never parses.
const PAYLOAD: u8 = 0xab;

/// One decimal JSON lexeme and the `u64` it denotes, if any.
#[derive(Debug, Clone)]
struct Lexeme {
    /// Exact JSON text.
    text: String,
    /// Parsed value when `text` is a canonical unsigned integer within `u64`.
    value: Option<u64>,
}

impl Lexeme {
    /// A canonical unsigned integer.
    fn integer(value: u64) -> Self {
        Self {
            text: value.to_string(),
            value: Some(value),
        }
    }

    /// Mostly `value`, occasionally a lexeme that is not a `u64`.
    fn perturbed(bytes: &mut Bytes<'_>, value: u64) -> Self {
        let invalid = |text: &str| Self {
            text: text.to_owned(),
            value: None,
        };
        match bytes.byte() {
            250 => invalid("-1"),
            251 => invalid("1.5"),
            252 => invalid("18446744073709551616"),
            253 => invalid("\"4\""),
            _ => Self::integer(value),
        }
    }
}

/// One generated header entry.
#[derive(Debug, Clone)]
struct TensorSpec {
    /// Tensor name (JSON key).
    name: String,
    /// Dtype spelling and bit width (zero when the spelling is invalid).
    dtype: (&'static str, u64),
    /// Dimension lexemes.
    shape: Vec<Lexeme>,
    /// Start and end lexemes.
    offsets: [Lexeme; 2],
}

impl TensorSpec {
    /// Exact byte length from the documented rule, or `None` when it is invalid.
    ///
    /// The element count is the running product of the dimensions; an
    /// intermediate overflow is rejected even when a later dimension is zero.
    fn byte_len(&self) -> Option<u64> {
        let mut elements = 1_u64;
        for dimension in &self.shape {
            elements = elements.checked_mul(dimension.value?)?;
        }
        let bits = self.dtype.1;
        if bits == 0 {
            return None;
        }
        let total = elements.checked_mul(bits)?;
        (total % 8 == 0).then_some(total / 8)
    }
}

/// One generated shard file.
#[derive(Debug, Clone)]
struct ShardSpec {
    /// Tensor entries in header order.
    tensors: Vec<TensorSpec>,
    /// Optional `__metadata__` entries, as raw `(key, JSON value)` pairs.
    metadata: Option<Vec<(String, String)>>,
    /// Index of the header entry before which `__metadata__` is written.
    metadata_position: usize,
    /// Trailing spaces after the closing brace.
    padding: usize,
    /// Data section length in bytes.
    data_len: u64,
    /// How the eight-byte length prefix relates to the real header length.
    prefix: Prefix,
}

/// Length-prefix perturbations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Prefix {
    /// The exact header length.
    Exact,
    /// Zero, which the format forbids.
    Zero,
    /// `u64::MAX`, beyond every limit and file.
    Maximum,
    /// One byte longer: the header would end inside the data section.
    Longer,
    /// One byte shorter: valid only when that byte is header padding.
    Shorter,
}

impl ShardSpec {
    /// The JSON header text, padding included.
    fn header(&self) -> String {
        let mut entries: Vec<String> = self
            .tensors
            .iter()
            .map(|tensor| {
                let shape: Vec<&str> = tensor.shape.iter().map(|d| d.text.as_str()).collect();
                format!(
                    "{}:{{\"dtype\":{},\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                    json_string(&tensor.name),
                    json_string(tensor.dtype.0),
                    shape.join(","),
                    tensor.offsets[0].text,
                    tensor.offsets[1].text
                )
            })
            .collect();
        if let Some(metadata) = &self.metadata {
            let fields: Vec<String> = metadata
                .iter()
                .map(|(key, value)| format!("{}:{value}", json_string(key)))
                .collect();
            let entry = format!("\"__metadata__\":{{{}}}", fields.join(","));
            entries.insert(self.metadata_position.min(entries.len()), entry);
        }
        format!("{{{}}}{}", entries.join(","), " ".repeat(self.padding))
    }

    /// The value written into the length prefix.
    fn prefix_value(&self) -> u64 {
        let actual = self.header().len() as u64;
        match self.prefix {
            Prefix::Exact => actual,
            Prefix::Zero => 0,
            Prefix::Maximum => u64::MAX,
            Prefix::Longer => actual + 1,
            Prefix::Shorter => actual.saturating_sub(1),
        }
    }

    /// The complete file: length prefix, header, and payload.
    fn file(&self) -> Vec<u8> {
        let header = self.header();
        let length = self.prefix_value();
        let mut file = length.to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.resize(
            file.len() + usize::try_from(self.data_len).expect("bounded data"),
            PAYLOAD,
        );
        file
    }
}

/// Serializes a JSON string literal.
fn json_string(text: &str) -> String {
    serde_json::to_string(text).expect("strings serialize")
}

/// An inventoried tensor as the oracle predicts it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedTensor {
    /// Dimensions.
    shape: Vec<u64>,
    /// Canonical dtype spelling.
    dtype: &'static str,
    /// Half-open data range.
    offsets: [u64; 2],
    /// Exact byte length.
    byte_len: u64,
}

/// A shard as the oracle predicts it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedShard {
    /// Tensors by name.
    tensors: BTreeMap<String, ExpectedTensor>,
    /// Metadata entries.
    metadata: BTreeMap<String, String>,
    /// Header length from the prefix.
    header_bytes: u64,
    /// Data section length.
    data_bytes: u64,
    /// Whole file length.
    file_bytes: u64,
}

/// Predicts the inventory of one shard, or `None` when discovery must reject it.
fn expected_shard(shard: &ShardSpec, limits: &ArtifactLimits) -> Option<ExpectedShard> {
    let header = shard.header();
    let actual = header.len() as u64;
    let file_bytes = 8 + actual + shard.data_len;
    let header_bytes = shard.prefix_value();
    if header_bytes > limits.max_header_bytes
        || header_bytes == 0
        || header_bytes.checked_add(8)? > file_bytes
    {
        return None;
    }
    // A shorter prefix stays valid JSON only by dropping trailing padding;
    // a longer one reads a payload byte, which is never JSON whitespace.
    if header_bytes > actual || actual - header_bytes > shard.padding as u64 {
        return None;
    }
    let data_bytes = file_bytes - 8 - header_bytes;
    let mut names = BTreeSet::new();
    for tensor in &shard.tensors {
        if !names.insert(tensor.name.as_str()) {
            return None;
        }
    }
    let mut metadata = BTreeMap::new();
    if let Some(entries) = &shard.metadata {
        for (key, value) in entries {
            let value: serde_json::Value = serde_json::from_str(value).expect("generated JSON");
            let text = value.as_str()?.to_owned();
            if metadata.insert(key.clone(), text).is_some() {
                return None;
            }
        }
        if metadata.len() > limits.max_metadata_entries {
            return None;
        }
    }
    if shard.tensors.len() > limits.max_tensors {
        return None;
    }
    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::new();
    for tensor in &shard.tensors {
        let shape: Vec<u64> = tensor
            .shape
            .iter()
            .map(|dimension| dimension.value)
            .collect::<Option<_>>()?;
        let start = tensor.offsets[0].value?;
        let end = tensor.offsets[1].value?;
        if tensor.name.is_empty() || shape.len() > limits.max_rank {
            return None;
        }
        let byte_len = tensor.byte_len()?;
        if end < start || end - start != byte_len || end > data_bytes {
            return None;
        }
        ranges.push((start, end, tensor.name.clone()));
        tensors.insert(
            tensor.name.clone(),
            ExpectedTensor {
                shape,
                dtype: tensor.dtype.0,
                offsets: [start, end],
                byte_len,
            },
        );
    }
    ranges.sort();
    let mut cursor = 0;
    for (start, end, _) in &ranges {
        if *start != cursor {
            return None;
        }
        cursor = *end;
    }
    (cursor == data_bytes).then_some(ExpectedShard {
        tensors,
        metadata,
        header_bytes,
        data_bytes,
        file_bytes,
    })
}

/// Decodes one header entry with the given name and a random dtype and shape.
fn tensor(bytes: &mut Bytes<'_>, name: String) -> TensorSpec {
    let dtype = bytes.pick(&DTYPES);
    let rank = bytes.below(5);
    let shape = (0..rank)
        .map(|_| {
            let dimension = bytes.pick(&DIMENSIONS);
            Lexeme::perturbed(bytes, dimension)
        })
        .collect();
    TensorSpec {
        name,
        dtype,
        shape,
        offsets: [Lexeme::integer(0), Lexeme::integer(0)],
    }
}

/// Decodes an MXFP4-style expert pair: packed blocks and their shared scales.
///
/// gpt-oss stores `[E, R, G, 16]` `U8` blocks (32 FP4 values per group) and
/// `[E, R, G]` `U8` scales; the microscaling dtypes describe the same layout as
/// `F4` values `[E, R, 32 G]` with `F8_E8M0` scales. An odd `F4` count is a
/// partial byte and must be rejected.
fn mxfp4_pair(bytes: &mut Bytes<'_>) -> [TensorSpec; 2] {
    let experts = bytes.pick(&[1, 2, 4]);
    let rows = bytes.pick(&[1, 3, 90]);
    let groups = bytes.pick(&[1, 2, 90]);
    let integer = Lexeme::integer;
    let packed = bytes.flag();
    let blocks = TensorSpec {
        name: "model.layers.0.mlp.experts.gate_up_proj_blocks".into(),
        dtype: if packed { ("U8", 8) } else { ("F4", 4) },
        shape: if packed {
            vec![
                integer(experts),
                integer(rows),
                integer(groups),
                integer(16),
            ]
        } else {
            let values = groups * 32 - u64::from(bytes.flag());
            vec![integer(experts), integer(rows), integer(values)]
        },
        offsets: [integer(0), integer(0)],
    };
    let scales = TensorSpec {
        name: "model.layers.0.mlp.experts.gate_up_proj_scales".into(),
        dtype: if packed { ("U8", 8) } else { ("F8_E8M0", 8) },
        shape: vec![integer(experts), integer(rows), integer(groups)],
        offsets: [integer(0), integer(0)],
    };
    [blocks, scales]
}

/// Assigns contiguous offsets in a decoded order, then perturbs a few.
///
/// Returns the covered byte count (capped at [`MAX_DATA`]).
fn assign_offsets(bytes: &mut Bytes<'_>, tensors: &mut [TensorSpec]) -> u64 {
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    for index in (1..order.len()).rev() {
        order.swap(index, bytes.below(index + 1));
    }
    let mut cursor = 0_u64;
    for &index in &order {
        let length = tensors[index].byte_len().unwrap_or(4).min(MAX_DATA);
        let start = cursor;
        cursor = cursor.saturating_add(length).min(MAX_DATA);
        tensors[index].offsets = [Lexeme::integer(start), Lexeme::integer(cursor)];
    }
    for tensor in tensors.iter_mut() {
        let [start, end] = [
            tensor.offsets[0].value.unwrap(),
            tensor.offsets[1].value.unwrap(),
        ];
        tensor.offsets = match bytes.byte() {
            240 => [Lexeme::integer(start + 1), Lexeme::integer(end)],
            241 => [Lexeme::integer(start), Lexeme::integer(end + 1)],
            242 => [Lexeme::integer(end), Lexeme::integer(start)],
            243 => [Lexeme::integer(start), Lexeme::integer(u64::MAX)],
            244 => [Lexeme::perturbed(bytes, start), Lexeme::integer(end)],
            _ => tensor.offsets.clone(),
        };
    }
    cursor
}

/// Decodes a shard: tensors (random or an MXFP4 pair), metadata, padding, and framing.
fn shard(bytes: &mut Bytes<'_>) -> ShardSpec {
    let mut tensors: Vec<TensorSpec> = if bytes.below(4) == 0 {
        mxfp4_pair(bytes).into()
    } else {
        (0..bytes.below(5))
            .map(|_| {
                let name = bytes.pick(&NAMES).to_owned();
                tensor(bytes, name)
            })
            .collect()
    };
    let covered = assign_offsets(bytes, &mut tensors);
    let metadata = bytes.flag().then(|| {
        (0..bytes.below(4))
            .map(|_| {
                let key = bytes.pick(&["format", "pt", "", "format"]).to_owned();
                let value = bytes.pick(&["\"pt\"", "\"\"", "1", "null"]).to_owned();
                (key, value)
            })
            .collect()
    });
    let metadata_position = bytes.below(tensors.len() + 1);
    let padding = bytes.below(4);
    let data_len = match bytes.byte() {
        250 => covered + 1,
        251 => covered.saturating_sub(1),
        _ => covered,
    };
    let prefix = match bytes.byte() {
        250 => Prefix::Zero,
        251 => Prefix::Maximum,
        252 => Prefix::Longer,
        253 | 254 => Prefix::Shorter,
        _ => Prefix::Exact,
    };
    ShardSpec {
        tensors,
        metadata,
        metadata_position,
        padding,
        data_len,
        prefix,
    }
}

/// Decodes artifact limits that are sometimes tight enough to bind.
fn limits(bytes: &mut Bytes<'_>) -> ArtifactLimits {
    ArtifactLimits {
        max_header_bytes: bytes.pick(&[64, 4096]),
        max_index_bytes: 8192,
        max_shards: bytes.pick(&[1, 4]),
        max_tensors: bytes.pick(&[2, 64]),
        max_rank: bytes.pick(&[3, 8]),
        max_metadata_entries: bytes.pick(&[1, 16]),
    }
}

/// Compares a discovered shard inventory with the oracle's prediction.
fn assert_shard_matches(
    shard: &transformers::checkpoint::ShardInventory,
    expected: &ExpectedShard,
) {
    assert_eq!(shard.header_bytes, expected.header_bytes);
    assert_eq!(shard.data_bytes, expected.data_bytes);
    assert_eq!(shard.file_bytes, expected.file_bytes);
    assert_eq!(shard.metadata, expected.metadata);
    assert_eq!(shard.tensors.len(), expected.tensors.len());
    for (name, tensor) in &shard.tensors {
        let predicted = &expected.tensors[name];
        assert_eq!(&tensor.name, name);
        assert_eq!(tensor.shape, predicted.shape);
        assert_eq!(tensor.dtype.as_str(), predicted.dtype);
        assert_eq!(tensor.data_offsets, predicted.offsets);
        assert_eq!(tensor.byte_len, predicted.byte_len);
        assert_eq!(
            shard.tensor_file_offset(tensor).unwrap(),
            8 + shard.header_bytes + predicted.offsets[0]
        );
    }
}

/// Checks a discovered single file against the official SafeTensors parser.
fn assert_official_agrees(file: &[u8], expected: &ExpectedShard) {
    let official = safetensors::SafeTensors::deserialize(file)
        .expect("the official parser accepts what discovery accepts");
    assert_eq!(official.len(), expected.tensors.len());
    for (name, tensor) in &expected.tensors {
        let view = official.tensor(name).expect("same tensor names");
        assert_eq!(view.dtype().to_string(), tensor.dtype);
        let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
        assert_eq!(shape, tensor.shape);
        assert_eq!(view.data().len() as u64, tensor.byte_len);
    }
}

/// Single-file mode: one generated shard against the oracle and the official parser.
///
/// Returns whether discovery accepted the file.
fn exercise_single(bytes: &mut Bytes<'_>) -> bool {
    let limits = limits(bytes);
    let spec = shard(bytes);
    let file = spec.file();
    let directory = tempfile::tempdir().expect("create isolated artifact directory");
    fs::write(directory.path().join("model.safetensors"), &file).expect("write shard");
    let selection = if bytes.flag() {
        ArtifactSelection::Auto
    } else {
        ArtifactSelection::Single("model.safetensors".into())
    };
    let options = LoadOptions {
        artifact: selection,
        limits: limits.clone(),
        ..LoadOptions::default()
    };
    let discovered = CheckpointManifest::discover(directory.path(), &options);
    let expected = expected_shard(&spec, &limits);
    match (&discovered, &expected) {
        (Ok(manifest), Some(expected)) => {
            assert_eq!(manifest.shards.len(), 1);
            assert_shard_matches(&manifest.shards[0], expected);
            assert_eq!(manifest.tensor_bytes, expected.data_bytes);
            assert_eq!(manifest.indexed_tensor_bytes, expected.data_bytes);
            assert_eq!(manifest.artifact_bytes, expected.file_bytes);
            assert_official_agrees(&file, expected);
            for (name, tensor) in &expected.tensors {
                let dtype: TensorDType = tensor.dtype.parse().expect("known dtype");
                assert_eq!(
                    dtype.tensor_storage_bytes(&tensor.shape).ok(),
                    Some(tensor.byte_len)
                );
                assert!(manifest.tensor(name).is_some());
            }
        }
        (Err(_), None) => {}
        _ => panic!(
            "discovery {:?} disagrees with the oracle {expected:?} for header {:?}",
            discovered.as_ref().map(|_| "accepted"),
            spec.header()
        ),
    }
    discovered.is_ok()
}

/// How the generated index deviates from the shards it describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexDefect {
    /// The weight map lists exactly the physical tensors.
    None,
    /// One tensor is missing from the weight map.
    Dropped,
    /// An entry names a tensor no header contains.
    Phantom,
    /// One tensor is assigned to another shard.
    Moved,
    /// Two shards contain a tensor of the same name.
    DuplicateTensor,
    /// An index-only alias of a tensor on the same shard.
    Alias,
    /// An index-only alias assigned to a different shard than its owner.
    CrossShardAlias,
    /// One shard name is unsafe, missing, or not a `.safetensors` file.
    UnsafeShard,
}

/// Writes up to three valid shards and an index with one defect; checks discovery.
///
/// Returns whether discovery accepted the index.
fn exercise_index(bytes: &mut Bytes<'_>) -> bool {
    let limits = limits(bytes);
    let directory = tempfile::tempdir().expect("create isolated artifact directory");
    let count = bytes.below(3) + 1;
    let defect = bytes.pick(&[
        IndexDefect::None,
        IndexDefect::None,
        IndexDefect::Dropped,
        IndexDefect::Phantom,
        IndexDefect::Moved,
        IndexDefect::DuplicateTensor,
        IndexDefect::Alias,
        IndexDefect::CrossShardAlias,
        IndexDefect::UnsafeShard,
    ]);
    // Physical tensors: shard file name -> (tensor name -> byte length).
    let mut shards: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    let mut file_bytes: BTreeMap<String, u64> = BTreeMap::new();
    let mut header_bytes: BTreeMap<String, u64> = BTreeMap::new();
    for shard_index in 0..count {
        let file_name = format!("{}.safetensors", char::from(b'a' + shard_index as u8));
        let mut tensors = Vec::new();
        for tensor_index in 0..bytes.below(3) + 1 {
            let name = if defect == IndexDefect::DuplicateTensor && tensor_index == 0 {
                "shared".to_owned()
            } else {
                format!("s{shard_index}.t{tensor_index}")
            };
            let mut spec = tensor(bytes, name);
            spec.dtype = bytes.pick(&DTYPES[..20]);
            spec.shape = (0..bytes.below(3))
                .map(|_| Lexeme::integer(bytes.pick(&[0, 1, 2, 4])))
                .collect();
            if spec.byte_len().is_none() {
                // An odd sub-byte count; make it whole bytes.
                spec.shape = vec![Lexeme::integer(8)];
            }
            tensors.push(spec);
        }
        let covered = assign_offsets(&mut Bytes::new(&[]), &mut tensors);
        let spec = ShardSpec {
            metadata: None,
            metadata_position: 0,
            padding: 0,
            data_len: covered,
            prefix: Prefix::Exact,
            tensors,
        };
        let file = spec.file();
        fs::write(directory.path().join(&file_name), &file).expect("write shard");
        file_bytes.insert(file_name.clone(), file.len() as u64);
        header_bytes.insert(file_name.clone(), spec.prefix_value());
        let lengths = spec
            .tensors
            .iter()
            .map(|tensor| (tensor.name.clone(), tensor.byte_len().unwrap()))
            .collect();
        shards.insert(file_name, lengths);
    }
    // The faithful weight map, then one defect.
    let mut weight_map: Vec<(String, String)> = shards
        .iter()
        .flat_map(|(shard, tensors)| tensors.keys().map(|name| (name.clone(), shard.clone())))
        .collect();
    if defect == IndexDefect::DuplicateTensor {
        // Index the shared name once, so the second shard's header repeats it.
        let mut seen = BTreeSet::new();
        weight_map.retain(|(name, _)| seen.insert(name.clone()));
    }
    let mut aliases = Vec::new();
    let shard_names: Vec<String> = shards.keys().cloned().collect();
    let target = bytes.below(weight_map.len());
    match defect {
        IndexDefect::None | IndexDefect::DuplicateTensor => {}
        IndexDefect::Dropped => {
            weight_map.remove(target);
        }
        IndexDefect::Phantom => weight_map.push(("phantom".into(), shard_names[0].clone())),
        IndexDefect::Moved => {
            let other = bytes.pick(&shard_names);
            weight_map[target].1 = other;
        }
        IndexDefect::Alias | IndexDefect::CrossShardAlias => {
            let (owner, shard) = weight_map[target].clone();
            let declared = if defect == IndexDefect::Alias {
                shard
            } else {
                bytes.pick(&shard_names)
            };
            weight_map.push(("alias.weight".into(), declared));
            if bytes.below(4) != 0 {
                aliases.push(IndexTensorAlias::new("alias.weight", owner));
            }
        }
        IndexDefect::UnsafeShard => {
            weight_map[target].1 = bytes
                .pick(&[
                    "../a.safetensors",
                    "/a.safetensors",
                    "a.bin",
                    "z.safetensors",
                    "",
                ])
                .to_owned();
        }
    }
    let accounting = if bytes.flag() {
        IndexAliasSizeAccounting::Physical
    } else {
        IndexAliasSizeAccounting::Logical
    };
    let total = expected_total(&shards, &weight_map, &aliases, accounting);
    let (total_text, total_value) = total_size(bytes, total);
    let entries: Vec<String> = weight_map
        .iter()
        .map(|(name, shard)| format!("{}:{}", json_string(name), json_string(shard)))
        .collect();
    let metadata = total_text.map_or(String::new(), |text| {
        format!("\"metadata\":{{\"total_size\":{text}}},")
    });
    let index = format!("{{{metadata}\"weight_map\":{{{}}}}}", entries.join(","));
    fs::write(
        directory.path().join("model.safetensors.index.json"),
        &index,
    )
    .expect("write index");
    let options = LoadOptions {
        artifact: ArtifactSelection::Index("model.safetensors.index.json".into()),
        limits: limits.clone(),
        index_aliases: aliases.clone(),
        index_alias_size_accounting: accounting,
        ..LoadOptions::default()
    };
    let discovered = CheckpointManifest::discover(directory.path(), &options);
    let expected = expected_index(
        &Shards {
            tensors: &shards,
            header_bytes: &header_bytes,
        },
        &weight_map,
        &aliases,
        accounting,
        total_value,
        &limits,
    );
    match (&discovered, &expected) {
        (Ok(manifest), Some(expected)) => {
            assert_eq!(manifest.tensor_bytes, expected.tensor_bytes);
            assert_eq!(manifest.indexed_tensor_bytes, expected.indexed_tensor_bytes);
            let artifact: u64 = expected.shards.iter().map(|shard| file_bytes[shard]).sum();
            assert_eq!(manifest.artifact_bytes, artifact);
            let discovered_shards: Vec<&str> = manifest
                .shards
                .iter()
                .map(|shard| shard.relative_path.as_str())
                .collect();
            let expected_shards: Vec<&str> = expected.shards.iter().map(String::as_str).collect();
            assert_eq!(discovered_shards, expected_shards);
            let tensors: BTreeSet<&str> = manifest.tensors.keys().map(String::as_str).collect();
            assert_eq!(
                tensors,
                expected.tensors.iter().map(String::as_str).collect()
            );
            assert_eq!(manifest.index_aliases.len(), expected.aliases);
        }
        (Err(_), None) => {}
        _ => panic!(
            "index discovery {:?} disagrees with the oracle {expected:?}: {index}",
            discovered.as_ref().map(|_| "accepted")
        ),
    }
    discovered.is_ok()
}

/// Logical or physical byte total the index should declare, if well defined.
fn expected_total(
    shards: &BTreeMap<String, BTreeMap<String, u64>>,
    weight_map: &[(String, String)],
    aliases: &[IndexTensorAlias],
    accounting: IndexAliasSizeAccounting,
) -> u64 {
    let physical: u64 = shards.values().flat_map(BTreeMap::values).sum();
    let alias_bytes: u64 = aliases
        .iter()
        .filter_map(|alias| {
            let (_, shard) = weight_map
                .iter()
                .find(|(name, _)| *name == alias.storage_key)?;
            shards.get(shard)?.get(&alias.storage_key).copied()
        })
        .sum();
    match accounting {
        IndexAliasSizeAccounting::Logical => physical + alias_bytes,
        _ => physical,
    }
}

/// Decodes a `metadata.total_size` lexeme around the exact total.
///
/// Returns the JSON text (absent when `None`) and the value discovery reads
/// from it: `Ok(None)` for absent or `null`, `Ok(Some)` for an accepted
/// integer, and `Err` for a lexeme that must be rejected.
fn total_size(bytes: &mut Bytes<'_>, total: u64) -> (Option<String>, Result<Option<u64>, ()>) {
    match bytes.below(10) {
        0 => (None, Ok(None)),
        1 => (Some("null".into()), Ok(None)),
        2 | 3 => (Some(total.to_string()), Ok(Some(total))),
        4 => (Some(format!("{total}.000")), Ok(Some(total))),
        5 => (Some((total + 1).to_string()), Ok(Some(total + 1))),
        6 => (Some(format!("{total}.5")), Err(())),
        7 => (Some(format!("-{total}")), Err(())),
        8 => (Some(format!("{total}e0")), Err(())),
        _ => (Some(format!("\"{total}\"")), Err(())),
    }
}

/// The physical shards written for an index problem.
struct Shards<'a> {
    /// Shard file name to its tensors' byte lengths.
    tensors: &'a BTreeMap<String, BTreeMap<String, u64>>,
    /// Shard file name to its JSON header length.
    header_bytes: &'a BTreeMap<String, u64>,
}

/// The oracle's prediction for an index discovery.
#[derive(Debug)]
struct ExpectedIndex {
    /// Referenced shard names in sorted order.
    shards: Vec<String>,
    /// Physical tensor names of referenced shards.
    tensors: BTreeSet<String>,
    /// Resolved alias count.
    aliases: usize,
    /// Sum of physical tensor bytes.
    tensor_bytes: u64,
    /// Total compared with `metadata.total_size`.
    indexed_tensor_bytes: u64,
}

/// Predicts index discovery from the documented reconciliation rules.
///
/// Only referenced shards are inspected, each within the header and tensor
/// limits; each header tensor must be indexed to
/// its own shard and appear in exactly one shard; each index entry must be a
/// physical tensor of its shard or an authorized alias whose storage owner is
/// indexed to, and present in, the same shard; the declared total must equal
/// the physical bytes plus, under logical accounting, each alias owner's bytes.
fn expected_index(
    shards: &Shards<'_>,
    weight_map: &[(String, String)],
    aliases: &[IndexTensorAlias],
    accounting: IndexAliasSizeAccounting,
    total: Result<Option<u64>, ()>,
    limits: &ArtifactLimits,
) -> Option<ExpectedIndex> {
    let declared = total.ok()?;
    let map: BTreeMap<&str, &str> = weight_map
        .iter()
        .map(|(name, shard)| (name.as_str(), shard.as_str()))
        .collect();
    if map.len() != weight_map.len() {
        return None;
    }
    let referenced: BTreeSet<&str> = map.values().copied().collect();
    if referenced.len() > limits.max_shards || map.len() > limits.max_tensors {
        return None;
    }
    let mut tensors: BTreeMap<&str, (&str, u64)> = BTreeMap::new();
    for &shard in &referenced {
        let physical = shards.tensors.get(shard)?;
        if physical.len() > limits.max_tensors
            || shards.header_bytes[shard] > limits.max_header_bytes
        {
            return None;
        }
        for (name, &length) in physical {
            if tensors.contains_key(name.as_str()) || map.get(name.as_str()) != Some(&shard) {
                return None;
            }
            tensors.insert(name, (shard, length));
        }
    }
    let authorized: BTreeMap<&str, &str> = aliases
        .iter()
        .map(|alias| (alias.alias.as_str(), alias.storage_key.as_str()))
        .collect();
    let mut alias_bytes = 0;
    let mut resolved = 0;
    for (&name, &shard) in &map {
        if tensors.contains_key(name) {
            continue;
        }
        let owner = authorized.get(name)?;
        let (owner_shard, length) = tensors.get(owner)?;
        if map.get(owner) != Some(&shard) || *owner_shard != shard {
            return None;
        }
        alias_bytes += length;
        resolved += 1;
    }
    let tensor_bytes: u64 = tensors.values().map(|(_, length)| length).sum();
    let indexed_tensor_bytes = match accounting {
        IndexAliasSizeAccounting::Logical => tensor_bytes + alias_bytes,
        _ => tensor_bytes,
    };
    if declared.is_some_and(|declared| declared != indexed_tensor_bytes) {
        return None;
    }
    Some(ExpectedIndex {
        shards: referenced.into_iter().map(str::to_owned).collect(),
        tensors: tensors.keys().map(|name| (*name).to_owned()).collect(),
        aliases: resolved,
        tensor_bytes,
        indexed_tensor_bytes,
    })
}

/// Decodes and checks one artifact problem.
///
/// Byte layout: a mode byte (below 160 selects a single file, otherwise an
/// index), the limits (header, shard, tensor, rank, and metadata bounds),
/// then the shard or index description. A shard is a tensor source (an
/// MXFP4-style pair one time in four, else up to four entries of
/// name/dtype/rank/dimensions), an offset permutation and per-tensor offset
/// perturbations, optional metadata, its position, padding, and data-length
/// and length-prefix perturbations. An index adds the shard count, the defect,
/// per-shard tensors, the defect's target, the accounting convention, and the
/// `total_size` spelling. Returns whether discovery accepted the artifact, so
/// deterministic tests can prove that both outcomes are exercised.
pub fn exercise(input: &[u8]) -> bool {
    let mut bytes = Bytes::new(input);
    if bytes.byte() < 160 {
        exercise_single(&mut bytes)
    } else {
        exercise_index(&mut bytes)
    }
}

#[cfg(test)]
#[path = "safetensors_headers_tests.rs"]
mod tests;
