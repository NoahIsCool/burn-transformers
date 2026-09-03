//! First-pass discovery of local SafeTensors artifacts.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{ErrorKind, Read},
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Deserializer, de};

use super::{ArtifactSelection, CheckpointError, LoadOptions, Result, TensorDType};

const STANDARD_INDEX: &str = "model.safetensors.index.json";
const STANDARD_SINGLE: &str = "model.safetensors";

/// Describes how a checkpoint manifest was selected.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArtifactSource {
    /// A Hugging Face shard index selected multiple tensor files.
    Index {
        /// Root-relative index filename.
        relative_path: String,
        /// Optional `metadata.total_size` declared by the index.
        declared_total_size: Option<u64>,
    },
    /// A single SafeTensors file was selected directly.
    Single {
        /// Root-relative tensor filename.
        relative_path: String,
    },
}

/// Validated metadata for one tensor in a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInventory {
    /// Globally unique checkpoint tensor name.
    pub name: String,
    /// Root-relative shard filename.
    pub shard: String,
    /// Tensor dimensions in logical row-major order.
    pub shape: Vec<u64>,
    /// Stored scalar dtype.
    pub dtype: TensorDType,
    /// Half-open byte range relative to the shard's data section.
    pub data_offsets: [u64; 2],
    /// Exact byte length derived from shape and dtype.
    pub byte_len: u64,
}

impl TensorInventory {
    /// Computes the checked number of logical scalar elements.
    pub fn elements(&self) -> Result<u64> {
        checked_elements(&self.shape)
    }
}

/// Validated metadata for one SafeTensors shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardInventory {
    /// Root-relative path recorded in the checkpoint manifest.
    pub relative_path: String,
    /// Canonical path proven to be below the canonical checkpoint root.
    pub path: PathBuf,
    /// Complete shard file size.
    pub file_bytes: u64,
    /// JSON header size, excluding the eight-byte length prefix.
    pub header_bytes: u64,
    /// Size of the validated tensor-data section.
    pub data_bytes: u64,
    /// Optional user metadata stored under `__metadata__`.
    pub metadata: BTreeMap<String, String>,
    /// Tensors ordered by name for reproducible planning and diagnostics.
    pub tensors: BTreeMap<String, TensorInventory>,
}

impl ShardInventory {
    /// Returns the absolute byte offset at which one tensor begins.
    pub fn tensor_file_offset(&self, tensor: &TensorInventory) -> Result<u64> {
        8_u64
            .checked_add(self.header_bytes)
            .and_then(|offset| offset.checked_add(tensor.data_offsets[0]))
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "computing tensor file offset",
            })
    }
}

/// Complete, cross-shard inventory produced without allocating tensor payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointManifest {
    /// Canonical repository directory.
    pub root: PathBuf,
    /// Artifact that selected the shards.
    pub source: ArtifactSource,
    /// Validated shards in root-relative filename order.
    pub shards: Vec<ShardInventory>,
    /// Globally unique tensors in name order.
    pub tensors: BTreeMap<String, TensorInventory>,
    /// Sum of validated tensor payload lengths.
    pub tensor_bytes: u64,
    /// Sum of complete shard file lengths.
    pub artifact_bytes: u64,
}

impl CheckpointManifest {
    /// Discovers and validates a checkpoint rooted at `root`.
    pub fn discover(root: impl AsRef<Path>, options: &LoadOptions) -> Result<Self> {
        discover_checkpoint(root, options)
    }

    /// Looks up a tensor by its checkpoint key.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorInventory> {
        self.tensors.get(name)
    }

    /// Looks up a shard by its root-relative filename.
    #[must_use]
    pub fn shard(&self, relative_path: &str) -> Option<&ShardInventory> {
        self.shards
            .iter()
            .find(|shard| shard.relative_path == relative_path)
    }
}

/// Discovers and fully validates local SafeTensors headers below `root`.
///
/// This function performs pass one of loading. It never reads tensor payload bytes;
/// each shard is held only while its bounded JSON header is decoded.
pub fn discover_checkpoint(
    root: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<CheckpointManifest> {
    let supplied_root = root.as_ref();
    let metadata = fs::metadata(supplied_root)
        .map_err(|error| CheckpointError::io("inspect checkpoint root", supplied_root, error))?;
    if !metadata.is_dir() {
        return Err(CheckpointError::RootNotDirectory {
            path: supplied_root.to_path_buf(),
        });
    }
    let root = fs::canonicalize(supplied_root).map_err(|error| {
        CheckpointError::io("canonicalize checkpoint root", supplied_root, error)
    })?;

    match &options.artifact {
        ArtifactSelection::Auto => discover_auto(root, options),
        ArtifactSelection::Index(path) => discover_index(root, path, options),
        ArtifactSelection::Single(path) => discover_single(root, path, options),
    }
}

fn discover_auto(root: PathBuf, options: &LoadOptions) -> Result<CheckpointManifest> {
    if root.join(STANDARD_INDEX).is_file() {
        return discover_index(root, STANDARD_INDEX, options);
    }
    if root.join(STANDARD_SINGLE).is_file() {
        return discover_single(root, STANDARD_SINGLE, options);
    }

    let mut indexes = Vec::new();
    let mut singles = Vec::new();
    let entries = fs::read_dir(&root)
        .map_err(|error| CheckpointError::io("list checkpoint root", &root, error))?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            CheckpointError::io("read checkpoint directory entry", &root, error)
        })?;
        let Some(name) = entry.file_name().to_str().map(ToOwned::to_owned) else {
            continue;
        };
        if name.ends_with(".safetensors.index.json") {
            indexes.push(name);
        } else if name.ends_with(".safetensors") {
            singles.push(name);
        }
    }
    indexes.sort();
    indexes.dedup();
    singles.sort();
    singles.dedup();

    match indexes.as_slice() {
        [only] => return discover_index(root, only, options),
        [] => {}
        _ => {
            return Err(CheckpointError::AmbiguousArtifacts {
                candidates: indexes,
            });
        }
    }
    match singles.as_slice() {
        [] => Err(CheckpointError::ArtifactNotFound { root }),
        [only] => discover_single(root, only, options),
        _ => Err(CheckpointError::AmbiguousArtifacts {
            candidates: singles,
        }),
    }
}

fn discover_single(
    root: PathBuf,
    relative_path: &str,
    options: &LoadOptions,
) -> Result<CheckpointManifest> {
    let path = resolve_local_file(&root, relative_path, "SafeTensors file")?;
    if !relative_path.ends_with(".safetensors") {
        return Err(CheckpointError::UnsafeRelativePath {
            path: relative_path.to_owned(),
            reason: "tensor shard must end in .safetensors",
        });
    }
    let shard = inspect_shard(relative_path, path, options)?;
    let tensor_bytes = shard.data_bytes;
    let artifact_bytes = shard.file_bytes;
    let tensors = shard.tensors.clone();
    Ok(CheckpointManifest {
        root,
        source: ArtifactSource::Single {
            relative_path: relative_path.to_owned(),
        },
        shards: vec![shard],
        tensors,
        tensor_bytes,
        artifact_bytes,
    })
}

fn discover_index(
    root: PathBuf,
    relative_path: &str,
    options: &LoadOptions,
) -> Result<CheckpointManifest> {
    let path = resolve_local_file(&root, relative_path, "checkpoint index")?;
    let index = read_index(&path, options)?;
    let shard_names: BTreeSet<_> = index.weight_map.values().cloned().collect();
    if shard_names.len() > options.limits.max_shards {
        return Err(limit("shard", shard_names.len(), options.limits.max_shards));
    }
    if index.weight_map.len() > options.limits.max_tensors {
        return Err(limit(
            "tensor",
            index.weight_map.len(),
            options.limits.max_tensors,
        ));
    }

    let mut shards = Vec::with_capacity(shard_names.len());
    let mut tensors: BTreeMap<String, TensorInventory> = BTreeMap::new();
    let mut artifact_bytes = 0_u64;
    let mut tensor_bytes = 0_u64;

    for shard_name in shard_names {
        if !shard_name.ends_with(".safetensors") {
            return Err(CheckpointError::UnsafeRelativePath {
                path: shard_name,
                reason: "tensor shard must end in .safetensors",
            });
        }
        let shard_path = resolve_local_file(&root, &shard_name, "checkpoint shard")?;
        let shard = inspect_shard(&shard_name, shard_path, options)?;
        artifact_bytes = artifact_bytes.checked_add(shard.file_bytes).ok_or(
            CheckpointError::ArithmeticOverflow {
                context: "summing checkpoint artifact bytes",
            },
        )?;
        for (name, tensor) in &shard.tensors {
            if let Some(previous) = tensors.get(name) {
                return Err(CheckpointError::DuplicateTensor {
                    tensor: name.clone(),
                    first_shard: previous.shard.clone(),
                    second_shard: shard_name.clone(),
                });
            }
            match index.weight_map.get(name) {
                Some(declared_shard) if declared_shard == &shard_name => {}
                Some(declared_shard) => {
                    return Err(CheckpointError::IndexMismatch {
                        tensor: name.clone(),
                        reason: format!(
                            "index assigns it to `{declared_shard}`, header is in `{shard_name}`"
                        ),
                    });
                }
                None => {
                    return Err(CheckpointError::IndexMismatch {
                        tensor: name.clone(),
                        reason: format!("present in `{shard_name}` but absent from weight_map"),
                    });
                }
            }
            tensors.insert(name.clone(), tensor.clone());
            tensor_bytes = tensor_bytes.checked_add(tensor.byte_len).ok_or(
                CheckpointError::ArithmeticOverflow {
                    context: "summing checkpoint tensor bytes",
                },
            )?;
        }
        shards.push(shard);
    }

    for (tensor, declared_shard) in &index.weight_map {
        if !tensors.contains_key(tensor) {
            return Err(CheckpointError::IndexMismatch {
                tensor: tensor.clone(),
                reason: format!("assigned to `{declared_shard}` but absent from its header"),
            });
        }
    }
    if let Some(declared) = index
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.total_size)
        && declared != tensor_bytes
    {
        return Err(CheckpointError::TotalSizeMismatch {
            declared,
            observed: tensor_bytes,
        });
    }

    Ok(CheckpointManifest {
        root,
        source: ArtifactSource::Index {
            relative_path: relative_path.to_owned(),
            declared_total_size: index.metadata.and_then(|metadata| metadata.total_size),
        },
        shards,
        tensors,
        tensor_bytes,
        artifact_bytes,
    })
}

fn resolve_local_file(root: &Path, untrusted: &str, kind: &'static str) -> Result<PathBuf> {
    validate_relative_path(untrusted)?;
    let joined = root.join(untrusted);
    let canonical = fs::canonicalize(&joined)
        .map_err(|error| CheckpointError::io("canonicalize checkpoint artifact", &joined, error))?;
    if !canonical.starts_with(root) {
        return Err(CheckpointError::PathEscapesRoot {
            root: root.to_path_buf(),
            path: canonical,
        });
    }
    let metadata = fs::metadata(&canonical)
        .map_err(|error| CheckpointError::io("inspect checkpoint artifact", &canonical, error))?;
    if !metadata.is_file() {
        return Err(CheckpointError::UnsafeRelativePath {
            path: untrusted.to_owned(),
            reason: if kind == "checkpoint shard" {
                "checkpoint shard is not a regular file"
            } else {
                "checkpoint artifact is not a regular file"
            },
        });
    }
    Ok(canonical)
}

fn validate_relative_path(path: &str) -> Result<()> {
    if path.is_empty() {
        return Err(unsafe_path(path, "path is empty"));
    }
    if path.contains('\0') {
        return Err(unsafe_path(path, "path contains a NUL byte"));
    }
    if path.contains('\\') {
        return Err(unsafe_path(
            path,
            "backslashes are forbidden in portable artifact paths",
        ));
    }
    let bytes = path.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return Err(unsafe_path(path, "Windows drive prefixes are forbidden"));
    }
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        return Err(unsafe_path(path, "absolute paths are forbidden"));
    }
    if candidate
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(unsafe_path(
            path,
            "only normal relative path components are permitted",
        ));
    }
    Ok(())
}

fn unsafe_path(path: &str, reason: &'static str) -> CheckpointError {
    CheckpointError::UnsafeRelativePath {
        path: path.to_owned(),
        reason,
    }
}

fn inspect_shard(
    relative_path: &str,
    path: PathBuf,
    options: &LoadOptions,
) -> Result<ShardInventory> {
    let mut file = File::open(&path)
        .map_err(|error| CheckpointError::io("open SafeTensors shard", &path, error))?;
    let file_bytes = file
        .metadata()
        .map_err(|error| CheckpointError::io("inspect SafeTensors shard", &path, error))?
        .len();
    if file_bytes < 8 {
        return Err(CheckpointError::TruncatedFile {
            path,
            size: file_bytes,
        });
    }

    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix).map_err(|error| {
        if error.kind() == ErrorKind::UnexpectedEof {
            CheckpointError::TruncatedFile {
                path: path.clone(),
                size: file_bytes,
            }
        } else {
            CheckpointError::io("read SafeTensors header length", &path, error)
        }
    })?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes > options.limits.max_header_bytes {
        return Err(CheckpointError::LimitExceeded {
            kind: "SafeTensors header byte",
            actual: header_bytes,
            limit: options.limits.max_header_bytes,
        });
    }
    let data_start =
        8_u64
            .checked_add(header_bytes)
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "computing SafeTensors data start",
            })?;
    if header_bytes == 0 || data_start > file_bytes {
        return Err(CheckpointError::InvalidHeaderLength {
            path,
            header_bytes,
            file_bytes,
        });
    }
    let header_len = usize::try_from(header_bytes).map_err(|_| CheckpointError::LimitExceeded {
        kind: "SafeTensors header byte",
        actual: header_bytes,
        limit: usize::MAX as u64,
    })?;
    let mut bytes = vec![0_u8; header_len];
    file.read_exact(&mut bytes)
        .map_err(|error| CheckpointError::io("read SafeTensors JSON header", &path, error))?;
    if bytes.first() != Some(&b'{') {
        return Err(CheckpointError::InvalidJson {
            kind: "SafeTensors header",
            path,
            message: "header must begin with `{`".to_owned(),
        });
    }
    let raw: RawHeader = serde_json::from_slice(&bytes)
        .map_err(|error| classify_json_error("SafeTensors header", &path, error))?;
    if raw.tensors.len() > options.limits.max_tensors {
        return Err(limit(
            "tensor",
            raw.tensors.len(),
            options.limits.max_tensors,
        ));
    }
    if raw.metadata.len() > options.limits.max_metadata_entries {
        return Err(limit(
            "SafeTensors metadata entry",
            raw.metadata.len(),
            options.limits.max_metadata_entries,
        ));
    }

    let data_bytes = file_bytes - data_start;
    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::with_capacity(raw.tensors.len());
    for (name, tensor) in raw.tensors {
        if name.is_empty() || name == "__metadata__" {
            return Err(CheckpointError::InvalidMapping {
                message: "tensor names must be nonempty and cannot be `__metadata__`".to_owned(),
            });
        }
        if tensor.shape.len() > options.limits.max_rank {
            return Err(limit(
                "tensor rank",
                tensor.shape.len(),
                options.limits.max_rank,
            ));
        }
        let dtype: TensorDType = tensor.dtype.parse()?;
        let elements = checked_elements(&tensor.shape)?;
        let byte_len = dtype.storage_bytes(elements)?;
        let [start, end] = tensor.data_offsets;
        let actual = end
            .checked_sub(start)
            .ok_or_else(|| CheckpointError::InvalidDataOffsets {
                tensor: name.clone(),
                shard: relative_path.to_owned(),
                reason: format!("end {end} precedes start {start}"),
            })?;
        if actual != byte_len {
            return Err(CheckpointError::TensorByteLengthMismatch {
                tensor: name,
                expected: byte_len,
                actual,
            });
        }
        if end > data_bytes {
            return Err(CheckpointError::InvalidDataOffsets {
                tensor: name,
                shard: relative_path.to_owned(),
                reason: format!("end {end} exceeds data section size {data_bytes}"),
            });
        }
        ranges.push((start, end, name.clone()));
        tensors.insert(
            name.clone(),
            TensorInventory {
                name,
                shard: relative_path.to_owned(),
                shape: tensor.shape,
                dtype,
                data_offsets: [start, end],
                byte_len,
            },
        );
    }
    ranges.sort_by(|left, right| (left.0, left.1, &left.2).cmp(&(right.0, right.1, &right.2)));
    let mut cursor = 0_u64;
    for (start, end, name) in ranges {
        if start != cursor {
            return Err(CheckpointError::InvalidDataOffsets {
                tensor: name,
                shard: relative_path.to_owned(),
                reason: format!("range starts at {start}, next contiguous byte is {cursor}"),
            });
        }
        cursor = end;
    }
    if cursor != data_bytes {
        return Err(CheckpointError::InvalidDataOffsets {
            tensor: "<end-of-data>".to_owned(),
            shard: relative_path.to_owned(),
            reason: format!("tensor ranges cover {cursor} bytes, data section has {data_bytes}"),
        });
    }

    // Run the format owner's validator as an independent compatibility check.
    // We parse only the bounded header rather than mapping the multi-gigabyte data
    // region; `Metadata` validates ordering, alignment, dtype spelling, and shapes.
    let official: safetensors::tensor::Metadata = serde_json::from_slice(&bytes)
        .map_err(|error| classify_json_error("SafeTensors header", &path, error))?;
    let official_data_bytes =
        u64::try_from(official.data_len()).map_err(|_| CheckpointError::ArithmeticOverflow {
            context: "converting SafeTensors metadata data length",
        })?;
    if official_data_bytes != data_bytes {
        return Err(CheckpointError::InvalidDataOffsets {
            tensor: "<end-of-data>".to_owned(),
            shard: relative_path.to_owned(),
            reason: format!(
                "official SafeTensors metadata covers {official_data_bytes} bytes, data section has {data_bytes}"
            ),
        });
    }

    Ok(ShardInventory {
        relative_path: relative_path.to_owned(),
        path,
        file_bytes,
        header_bytes,
        data_bytes,
        metadata: raw.metadata,
        tensors,
    })
}

fn checked_elements(shape: &[u64]) -> Result<u64> {
    shape.iter().try_fold(1_u64, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .ok_or(CheckpointError::ArithmeticOverflow {
                context: "computing tensor element count",
            })
    })
}

fn limit(kind: &'static str, actual: usize, configured: usize) -> CheckpointError {
    CheckpointError::LimitExceeded {
        kind,
        actual: actual as u64,
        limit: configured as u64,
    }
}

#[derive(Debug, Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Debug, Default)]
struct RawHeader {
    metadata: BTreeMap<String, String>,
    tensors: BTreeMap<String, RawTensor>,
}

impl<'de> Deserialize<'de> for RawHeader {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HeaderVisitor;

        impl<'de> de::Visitor<'de> for HeaderVisitor {
            type Value = RawHeader;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a SafeTensors header object")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut header = RawHeader::default();
                let mut seen = BTreeSet::new();
                while let Some(key) = map.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(de::Error::custom(format!(
                            "duplicate top-level key `{key}`"
                        )));
                    }
                    if key == "__metadata__" {
                        header.metadata = map.next_value::<UniqueStringMap>()?.0;
                    } else {
                        header.tensors.insert(key, map.next_value()?);
                    }
                }
                Ok(header)
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

#[derive(Debug, Default)]
struct UniqueStringMap(BTreeMap<String, String>);

impl<'de> Deserialize<'de> for UniqueStringMap {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StringMapVisitor;

        impl<'de> de::Visitor<'de> for StringMapVisitor {
            type Value = UniqueStringMap;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an object with unique string keys and string values")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, String>()? {
                    if values.insert(key.clone(), value).is_some() {
                        return Err(de::Error::custom(format!("duplicate map key `{key}`")));
                    }
                }
                Ok(UniqueStringMap(values))
            }
        }

        deserializer.deserialize_map(StringMapVisitor)
    }
}

#[derive(Debug, Deserialize)]
struct RawIndex {
    #[serde(default)]
    metadata: Option<RawIndexMetadata>,
    #[serde(deserialize_with = "deserialize_unique_string_map")]
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawIndexMetadata {
    #[serde(default)]
    total_size: Option<u64>,
}

fn deserialize_unique_string_map<'de, D>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, String>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(UniqueStringMap::deserialize(deserializer)?.0)
}

fn read_index(path: &Path, options: &LoadOptions) -> Result<RawIndex> {
    let file = File::open(path)
        .map_err(|error| CheckpointError::io("open checkpoint index", path, error))?;
    let size = file
        .metadata()
        .map_err(|error| CheckpointError::io("inspect checkpoint index", path, error))?
        .len();
    if size > options.limits.max_index_bytes {
        return Err(CheckpointError::LimitExceeded {
            kind: "checkpoint index byte",
            actual: size,
            limit: options.limits.max_index_bytes,
        });
    }
    // `take(limit + 1)` preserves the allocation bound if another local process
    // appends to the file between metadata inspection and reading it.
    let read_limit = options.limits.max_index_bytes.saturating_add(1);
    let mut bytes = Vec::new();
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| CheckpointError::io("read checkpoint index", path, error))?;
    if bytes.len() as u64 > options.limits.max_index_bytes {
        return Err(CheckpointError::LimitExceeded {
            kind: "checkpoint index byte",
            actual: bytes.len() as u64,
            limit: options.limits.max_index_bytes,
        });
    }
    serde_json::from_slice(&bytes)
        .map_err(|error| classify_json_error("checkpoint index", path, error))
}

fn classify_json_error(
    kind: &'static str,
    path: &Path,
    error: serde_json::Error,
) -> CheckpointError {
    let message = error.to_string();
    if message.contains("duplicate")
        && let Some(key) = message.split('`').nth(1)
    {
        return CheckpointError::DuplicateJsonKey {
            kind,
            path: path.to_path_buf(),
            key: key.to_owned(),
        };
    }
    CheckpointError::InvalidJson {
        kind,
        path: path.to_path_buf(),
        message,
    }
}
