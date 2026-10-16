//! First-pass discovery of local SafeTensors artifacts.
//!
//! Discovery reads JSON metadata, never model payloads. A tensor with dimensions
//! `d[0..rank]` must own exactly `product(d) * dtype_bits / 8` bytes in a
//! little-endian, row-major data section. Checked arithmetic, duplicate-key
//! rejection, and contiguous non-overlapping ranges make the inventory safe to
//! consume before expensive backend allocation.
//!
//! Index paths remain inside the canonical local root. Index aliases and defective
//! byte-total exceptions require explicit architecture policy; matching names or
//! total element counts alone never establish compatible tensor wiring.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{ErrorKind, Read},
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Deserializer, Serialize, de};
use sha2::{Digest, Sha256};

use super::{
    ArtifactSelection, CheckpointError, IndexAliasSizeAccounting, IndexTensorAlias, LoadOptions,
    Result, TensorDType,
    accounting::{IndexTotalSizeEvidence, PinnedControlFile, PinnedIndexTotalSizePolicy},
    layout::{
        DataSectionError, TilingError, data_section, element_count, tensor_file_offset,
        validate_contiguous_tiling,
    },
};

/// Preferred Hugging Face index filename during automatic discovery.
const STANDARD_INDEX: &str = "model.safetensors.index.json";
/// Preferred unsharded filename when no standard index exists.
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

/// One index-only alias resolved to physical storage during header discovery.
///
/// Unlike [`IndexTensorAlias`], which is architecture policy and therefore does
/// not know artifact placement, this record preserves the shard assignment
/// proven by the index and the storage owner's validated header. Keeping that
/// provenance in the manifest lets planning recheck public manifest values
/// before it emits a strict load report.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ResolvedIndexTensorAlias {
    /// Index key omitted from its assigned shard header.
    pub alias: String,
    /// Physical tensor key whose payload owns the shared storage.
    pub storage_key: String,
    /// Root-relative shard assigned to both keys by the checkpoint index.
    pub shard: String,
}

impl ResolvedIndexTensorAlias {
    /// Creates a resolved alias carrying its validated same-shard provenance.
    #[must_use]
    pub fn new(
        alias: impl Into<String>,
        storage_key: impl Into<String>,
        shard: impl Into<String>,
    ) -> Self {
        Self {
            alias: alias.into(),
            storage_key: storage_key.into(),
            shard: shard.into(),
        }
    }
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
        tensor_file_offset(self.header_bytes, tensor.data_offsets[0]).ok_or(
            CheckpointError::ArithmeticOverflow {
                context: "computing tensor file offset",
            },
        )
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
    /// Validated index-only aliases resolved to physical same-shard storage.
    pub index_aliases: Vec<ResolvedIndexTensorAlias>,
    /// Alias-size convention proven against an index, or `None` for one file.
    pub index_alias_size_accounting: Option<IndexAliasSizeAccounting>,
    /// Byte total validated against index metadata under the recorded convention.
    ///
    /// This equals [`Self::tensor_bytes`] for unsharded artifacts and indexes
    /// using physical accounting. Logical accounting additionally counts the
    /// storage-owner byte length once for every resolved index-only alias.
    pub indexed_tensor_bytes: u64,
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
    discover_checkpoint_with_policy(root, options, None)
}

/// Discovers a checkpoint with one architecture-owned total-size policy.
///
/// This crate-private entry point is intentionally absent from the public
/// loader API. Architectures may use it only with an immutable, content-bound
/// policy; ordinary [`CheckpointManifest::discover`] always remains strict.
pub(crate) fn discover_checkpoint_with_index_total_size_policy(
    root: impl AsRef<Path>,
    options: &LoadOptions,
    policy: &PinnedIndexTotalSizePolicy,
) -> Result<CheckpointManifest> {
    discover_checkpoint_with_policy(root, options, Some(policy))
}

/// Shared discovery implementation with an optional pinned accounting policy.
fn discover_checkpoint_with_policy(
    root: impl AsRef<Path>,
    options: &LoadOptions,
    policy: Option<&PinnedIndexTotalSizePolicy>,
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
        ArtifactSelection::Auto => discover_auto(root, options, policy),
        ArtifactSelection::Index(path) => discover_index(root, path, options, policy),
        ArtifactSelection::Single(path) => discover_single(root, path, options),
    }
}

// Artifact selection and index/shard reconciliation.

/// Prefers standard artifact names, then accepts only an unambiguous fallback.
///
/// When both `model.safetensors` and `model.safetensors.index.json` exist,
/// Hugging Face loads the single file and ignores the index. Discovery accepts
/// that layout only when the two provably agree: the index parses and its
/// `weight_map` names exactly the single file, as some exported repositories
/// ship. It then loads through the index, which validates every listed tensor
/// against that file. An index that names other shards, names none, or cannot
/// be read could describe different weights, so discovery fails closed with
/// [`CheckpointError::AmbiguousArtifacts`] and the caller selects an artifact
/// explicitly (an explicit index selection reports the precise index error).
/// The selected artifact still passes the same containment and metadata checks
/// as an explicitly named file; directory enumeration does not authorize its
/// paths.
fn discover_auto(
    root: PathBuf,
    options: &LoadOptions,
    policy: Option<&PinnedIndexTotalSizePolicy>,
) -> Result<CheckpointManifest> {
    let has_index = root.join(STANDARD_INDEX).is_file();
    let has_single = root.join(STANDARD_SINGLE).is_file();
    match (has_index, has_single) {
        (true, true) => {
            return match read_single_file_index(&root, options) {
                Some(document) => {
                    discover_index_document(root, STANDARD_INDEX, document, options, policy)
                }
                None => Err(CheckpointError::AmbiguousArtifacts {
                    candidates: vec![STANDARD_SINGLE.to_owned(), STANDARD_INDEX.to_owned()],
                }),
            };
        }
        (true, false) => return discover_index(root, STANDARD_INDEX, options, policy),
        (false, true) => return discover_single(root, STANDARD_SINGLE, options),
        (false, false) => {}
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
        [only] => return discover_index(root, only, options, policy),
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

/// Inventories one contained SafeTensors file and derives totals from its header.
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
        index_aliases: Vec::new(),
        index_alias_size_accounting: None,
        indexed_tensor_bytes: tensor_bytes,
        tensor_bytes,
        artifact_bytes,
    })
}

/// Returns the standard index when its `weight_map` names only `model.safetensors`.
///
/// Any read, size-limit, or parse failure, and any other shard set (including
/// an empty one), yields `None`: the two standard artifacts are then not
/// provably the same weights.
fn read_single_file_index(root: &Path, options: &LoadOptions) -> Option<RawIndexDocument> {
    let path = resolve_local_file(root, STANDARD_INDEX, "checkpoint index").ok()?;
    let document = read_index(&path, options).ok()?;
    let weight_map = &document.index.weight_map;
    let names_only_the_single_file =
        !weight_map.is_empty() && weight_map.values().all(|shard| shard == STANDARD_SINGLE);
    names_only_the_single_file.then_some(document)
}

/// Reads the selected index file and cross-checks it against its shards.
///
/// See [`discover_index_document`] for the reconciliation rules.
fn discover_index(
    root: PathBuf,
    relative_path: &str,
    options: &LoadOptions,
    policy: Option<&PinnedIndexTotalSizePolicy>,
) -> Result<CheckpointManifest> {
    let path = resolve_local_file(&root, relative_path, "checkpoint index")?;
    let index_document = read_index(&path, options)?;
    discover_index_document(root, relative_path, index_document, options, policy)
}

/// Cross-checks every index entry against all shard inventories before returning.
///
/// Physical tensor names must be unique and assigned to their declared shards.
/// Missing aliases and byte-total exceptions require the caller's explicit policy;
/// no tensor payload is materialized during this pass.
///
/// Shard names in `weight_map` are relative to the index's own directory, as in
/// Hugging Face's loader, so an index selected below the root (for example
/// `text_encoder/model.safetensors.index.json`) finds its co-located shards.
/// Each name is validated as a portable relative path before it is joined, and
/// the joined root-relative path identifies the shard everywhere downstream.
fn discover_index_document(
    root: PathBuf,
    relative_path: &str,
    index_document: RawIndexDocument,
    options: &LoadOptions,
    policy: Option<&PinnedIndexTotalSizePolicy>,
) -> Result<CheckpointManifest> {
    let index = &index_document.index;
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
    let authorized_aliases = validated_index_alias_policy(&options.index_aliases)?;

    let mut shards = Vec::with_capacity(shard_names.len());
    let mut tensors: BTreeMap<String, TensorInventory> = BTreeMap::new();
    let mut artifact_bytes = 0_u64;
    let mut tensor_bytes = 0_u64;

    let index_directory = relative_path
        .rsplit_once('/')
        .map(|(directory, _)| directory);
    let mut shard_paths = BTreeMap::new();
    for shard_name in shard_names {
        if !shard_name.ends_with(".safetensors") {
            return Err(CheckpointError::UnsafeRelativePath {
                path: shard_name,
                reason: "tensor shard must end in .safetensors",
            });
        }
        validate_relative_path(&shard_name)?;
        let shard_relative = match index_directory {
            Some(directory) => format!("{directory}/{shard_name}"),
            None => shard_name.clone(),
        };
        let shard_path = resolve_local_file(&root, &shard_relative, "checkpoint shard")?;
        let shard = inspect_shard(&shard_relative, shard_path, options)?;
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
                    second_shard: shard_relative.clone(),
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
        shard_paths.insert(shard_name, shard_relative);
    }

    let mut resolved_aliases = Vec::new();
    for (tensor, declared_shard) in &index.weight_map {
        if !tensors.contains_key(tensor) {
            let Some(storage_key) = authorized_aliases.get(tensor) else {
                return Err(CheckpointError::IndexMismatch {
                    tensor: tensor.clone(),
                    reason: format!("assigned to `{declared_shard}` but absent from its header"),
                });
            };
            let Some(storage_shard) = index.weight_map.get(storage_key) else {
                return Err(CheckpointError::InvalidIndexAlias {
                    alias: tensor.clone(),
                    reason: format!("storage key `{storage_key}` is absent from weight_map"),
                });
            };
            if storage_shard != declared_shard {
                return Err(CheckpointError::InvalidIndexAlias {
                    alias: tensor.clone(),
                    reason: format!(
                        "alias is assigned to `{declared_shard}` but storage key `{storage_key}` is assigned to `{storage_shard}`"
                    ),
                });
            }
            let Some(storage) = tensors.get(storage_key) else {
                return Err(CheckpointError::InvalidIndexAlias {
                    alias: tensor.clone(),
                    reason: format!(
                        "storage key `{storage_key}` is not present in its assigned shard header"
                    ),
                });
            };
            let declared_relative = shard_paths
                .get(declared_shard)
                .map_or(declared_shard.as_str(), String::as_str);
            if storage.shard != declared_relative {
                return Err(CheckpointError::InvalidIndexAlias {
                    alias: tensor.clone(),
                    reason: format!(
                        "validated storage key `{storage_key}` belongs to `{}`, not `{declared_relative}`",
                        storage.shard
                    ),
                });
            }
            resolved_aliases.push(ResolvedIndexTensorAlias::new(
                tensor,
                storage_key,
                declared_relative,
            ));
        }
    }
    resolved_aliases.sort();
    let indexed_tensor_bytes = match options.index_alias_size_accounting {
        IndexAliasSizeAccounting::Logical => {
            resolved_aliases
                .iter()
                .try_fold(tensor_bytes, |total, alias| {
                    let storage = tensors.get(&alias.storage_key).ok_or_else(|| {
                    CheckpointError::InvalidIndexAlias {
                        alias: alias.alias.clone(),
                        reason: format!(
                            "validated storage key `{}` disappeared from the immutable manifest",
                            alias.storage_key
                        ),
                    }
                })?;
                    total
                        .checked_add(storage.byte_len)
                        .ok_or(CheckpointError::ArithmeticOverflow {
                            context: "summing logical index tensor bytes including aliases",
                        })
                })?
        }
        IndexAliasSizeAccounting::Physical => tensor_bytes,
    };
    let declared_total_size = index
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.total_size);
    if let Some(declared) = declared_total_size
        && declared != indexed_tensor_bytes
    {
        let Some(policy) = policy else {
            return Err(CheckpointError::TotalSizeMismatch {
                declared,
                observed: indexed_tensor_bytes,
            });
        };
        validate_discovered_total_size_policy(
            &root,
            relative_path,
            &index_document,
            declared,
            tensor_bytes,
            &tensors,
            resolved_aliases.len(),
            policy,
        )?;
    }

    Ok(CheckpointManifest {
        root,
        source: ArtifactSource::Index {
            relative_path: relative_path.to_owned(),
            declared_total_size,
        },
        shards,
        tensors,
        index_aliases: resolved_aliases,
        index_alias_size_accounting: Some(options.index_alias_size_accounting),
        indexed_tensor_bytes,
        tensor_bytes,
        artifact_bytes,
    })
}

/// Authenticates and validates a policy against freshly discovered evidence.
#[allow(clippy::too_many_arguments)]
fn validate_discovered_total_size_policy(
    root: &Path,
    index_relative_path: &str,
    index: &RawIndexDocument,
    declared_total_size: u64,
    physical_tensor_bytes: u64,
    tensors: &BTreeMap<String, TensorInventory>,
    index_alias_count: usize,
    policy: &PinnedIndexTotalSizePolicy,
) -> Result<()> {
    let (config_bytes, config_sha256) = read_pinned_control_identity(root, &policy.config)?;
    policy.validate(IndexTotalSizeEvidence {
        index_relative_path,
        index_bytes: index.bytes,
        index_sha256: &index.sha256,
        config_relative_path: &policy.config.relative_path,
        config_bytes,
        config_sha256: &config_sha256,
        declared_total_size,
        physical_tensor_bytes,
        tensors,
        index_alias_count,
    })
}

/// Revalidates a public manifest against the same policy used at discovery.
///
/// [`CheckpointManifest`] exposes its inventories for diagnostics, so planning
/// must not trust that callers preserved them after discovery. This function
/// re-authenticates both control files and the complete implicated tensor set.
pub(super) fn validate_manifest_index_total_size_policy(
    manifest: &CheckpointManifest,
    policy: &PinnedIndexTotalSizePolicy,
) -> Result<()> {
    let (index_relative_path, declared_total_size) = match &manifest.source {
        ArtifactSource::Index {
            relative_path,
            declared_total_size: Some(declared),
        } => (relative_path.as_str(), *declared),
        ArtifactSource::Index {
            declared_total_size: None,
            ..
        } => {
            return Err(CheckpointError::IndexTotalSizePolicyMismatch {
                reason: "pinned accounting requires index metadata.total_size".into(),
            });
        }
        ArtifactSource::Single { .. } => {
            return Err(CheckpointError::IndexTotalSizePolicyMismatch {
                reason: "pinned index accounting cannot apply to an unsharded artifact".into(),
            });
        }
    };
    let observed_index = PinnedControlFile {
        relative_path: index_relative_path.to_owned(),
        ..policy.index.clone()
    };
    let (index_bytes, index_sha256) =
        read_pinned_control_identity(&manifest.root, &observed_index)?;
    let (config_bytes, config_sha256) =
        read_pinned_control_identity(&manifest.root, &policy.config)?;
    policy.validate(IndexTotalSizeEvidence {
        index_relative_path,
        index_bytes,
        index_sha256: &index_sha256,
        config_relative_path: &policy.config.relative_path,
        config_bytes,
        config_sha256: &config_sha256,
        declared_total_size,
        physical_tensor_bytes: manifest.tensor_bytes,
        tensors: &manifest.tensors,
        index_alias_count: manifest.index_aliases.len(),
    })
}

/// Reads and hashes one small pinned control file under a fixed allocation bound.
fn read_pinned_control_identity(
    root: &Path,
    expected: &PinnedControlFile,
) -> Result<(u64, String)> {
    let path = resolve_local_file(root, &expected.relative_path, "checkpoint control file")?;
    let file = File::open(&path)
        .map_err(|error| CheckpointError::io("open checkpoint control file", &path, error))?;
    let read_limit = expected.bytes.saturating_add(1);
    let capacity = usize::try_from(read_limit).map_err(|_| CheckpointError::LimitExceeded {
        kind: "pinned control-file byte",
        actual: read_limit,
        limit: usize::MAX as u64,
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| CheckpointError::io("read checkpoint control file", &path, error))?;
    let observed_bytes = bytes.len() as u64;
    Ok((observed_bytes, format!("{:x}", Sha256::digest(&bytes))))
}

/// Validates exact index-alias policy and returns a deterministic lookup table.
pub(super) fn validated_index_alias_policy(
    aliases: &[IndexTensorAlias],
) -> Result<BTreeMap<String, String>> {
    let mut validated = BTreeMap::new();
    for alias in aliases {
        if alias.alias.is_empty() {
            return Err(CheckpointError::InvalidIndexAlias {
                alias: alias.alias.clone(),
                reason: "alias key must not be empty".into(),
            });
        }
        if alias.storage_key.is_empty() {
            return Err(CheckpointError::InvalidIndexAlias {
                alias: alias.alias.clone(),
                reason: "storage key must not be empty".into(),
            });
        }
        if alias.alias == alias.storage_key {
            return Err(CheckpointError::InvalidIndexAlias {
                alias: alias.alias.clone(),
                reason: "alias and storage key must differ".into(),
            });
        }
        if let Some(previous) = validated.insert(alias.alias.clone(), alias.storage_key.clone())
            && previous != alias.storage_key
        {
            return Err(CheckpointError::InvalidIndexAlias {
                alias: alias.alias.clone(),
                reason: format!(
                    "alias maps to both `{previous}` and `{}`",
                    alias.storage_key
                ),
            });
        }
    }
    Ok(validated)
}

// Portable paths and bounded SafeTensors metadata validation.

/// Resolves an untrusted relative filename to a regular file inside the root.
///
/// Canonical containment rejects symlink escapes after the portable path check.
/// The runtime source repeats containment checks when opening payloads.
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

/// Rejects absolute, parent, drive-prefixed, NUL, and nonportable artifact paths.
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

/// Attaches the original untrusted path to a precise path-validation failure.
fn unsafe_path(path: &str, reason: &'static str) -> CheckpointError {
    CheckpointError::UnsafeRelativePath {
        path: path.to_owned(),
        reason,
    }
}

/// Reads a bounded shard header and proves exact contiguous coverage of its data.
///
/// For each tensor, the byte interval must equal `product(shape) * dtype_bits / 8`
/// without overflow or partial bytes. Sorted intervals must neither overlap nor
/// leave gaps, including trailing bytes. The official SafeTensors metadata parser
/// then independently validates the same header without reading weight payloads.
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
    // Verified: the section starts after the prefix and header and ends at the file's end.
    let section = match data_section(header_bytes, file_bytes) {
        Ok(section) => section,
        Err(DataSectionError::Overflow) => {
            return Err(CheckpointError::ArithmeticOverflow {
                context: "computing SafeTensors data start",
            });
        }
        Err(DataSectionError::InvalidHeaderLength) => {
            return Err(CheckpointError::InvalidHeaderLength {
                path,
                header_bytes,
                file_bytes,
            });
        }
    };
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

    let data_bytes = section.bytes;
    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::with_capacity(raw.tensors.len());
    for (name, tensor) in raw.tensors {
        if name.is_empty() || name == "__metadata__" {
            return Err(CheckpointError::InvalidTensorName {
                shard: relative_path.to_owned(),
                tensor: name,
                reason: "tensor names must be nonempty and cannot be `__metadata__`",
            });
        }
        if tensor.shape.len() > options.limits.max_rank {
            return Err(limit(
                "tensor rank",
                tensor.shape.len(),
                options.limits.max_rank,
            ));
        }
        let (dtype, byte_len) = describe_header_entry(&tensor)
            .map_err(|error| invalid_header_entry(relative_path, &name, error))?;
        let [start, end] = tensor.data_offsets;
        let actual = end
            .checked_sub(start)
            .ok_or_else(|| CheckpointError::InvalidDataOffsets {
                tensor: name.clone(),
                shard: relative_path.to_owned(),
                reason: format!("end {end} precedes start {start}"),
            })?;
        if actual != byte_len {
            let mismatch = CheckpointError::TensorByteLengthMismatch {
                tensor: name.clone(),
                expected: byte_len,
                actual,
            };
            return Err(invalid_header_entry(relative_path, &name, mismatch));
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
    let intervals = ranges
        .iter()
        .map(|(start, end, _)| (*start, *end))
        .collect::<Vec<_>>();
    if let Err(error) = validate_contiguous_tiling(&intervals, data_bytes) {
        let (tensor, reason) = match error {
            TilingError::Reversed { index } => (
                ranges[index].2.clone(),
                format!("end {} precedes start {}", ranges[index].1, ranges[index].0),
            ),
            TilingError::Discontiguous { index, expected } => (
                ranges[index].2.clone(),
                format!(
                    "range starts at {}, next contiguous byte is {expected}",
                    ranges[index].0
                ),
            ),
            TilingError::OutOfBounds { index } => (
                ranges[index].2.clone(),
                format!(
                    "end {} exceeds data section size {data_bytes}",
                    ranges[index].1
                ),
            ),
            TilingError::Uncovered { covered } => (
                "<end-of-data>".to_owned(),
                format!("tensor ranges cover {covered} bytes, data section has {data_bytes}"),
            ),
        };
        return Err(CheckpointError::InvalidDataOffsets {
            tensor,
            shard: relative_path.to_owned(),
            reason,
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

/// Parses one header entry's dtype and derives its exact byte length.
///
/// The length is `product(shape) * bits / 8`; overflow and partial bytes fail.
fn describe_header_entry(tensor: &RawTensor) -> Result<(TensorDType, u64)> {
    let dtype: TensorDType = tensor.dtype.parse()?;
    let byte_len = dtype.storage_bytes(checked_elements(&tensor.shape)?)?;
    Ok((dtype, byte_len))
}

/// Attaches shard and tensor context to one header-entry validation failure.
fn invalid_header_entry(shard: &str, tensor: &str, error: CheckpointError) -> CheckpointError {
    CheckpointError::InvalidTensorHeader {
        shard: shard.to_owned(),
        tensor: tensor.to_owned(),
        source: Box::new(error),
    }
}

/// Multiplies dimensions with checked arithmetic; a rank-zero tensor has one scalar.
fn checked_elements(shape: &[u64]) -> Result<u64> {
    element_count(shape).ok_or(CheckpointError::ArithmeticOverflow {
        context: "computing tensor element count",
    })
}

/// Reports an exceeded metadata count bound in the common unsigned-byte/count form.
fn limit(kind: &'static str, actual: usize, configured: usize) -> CheckpointError {
    CheckpointError::LimitExceeded {
        kind,
        actual: actual as u64,
        limit: configured as u64,
    }
}

// Serde visitors preserve exact key and integer semantics at the trust boundary.

/// Unvalidated SafeTensors header entry, with exact integer dimensions and offsets.
#[derive(Debug, Deserialize)]
struct RawTensor {
    /// Wire dtype spelling, validated before calculating storage size.
    dtype: String,
    /// Dimensions before checked conversion to host indexing sizes.
    shape: Vec<u64>,
    /// Half-open payload byte range relative to the data section.
    data_offsets: [u64; 2],
}

/// Header maps after duplicate decoded keys have been rejected.
#[derive(Debug, Default)]
struct RawHeader {
    /// User metadata separated from tensor descriptors.
    metadata: BTreeMap<String, String>,
    /// Tensor descriptors ordered by their decoded names.
    tensors: BTreeMap<String, RawTensor>,
}

impl<'de> Deserialize<'de> for RawHeader {
    /// Deserializes a SafeTensors header with duplicate decoded keys rejected.
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        /// Rejects duplicate keys while separating metadata from tensor entries.
        struct HeaderVisitor;

        impl<'de> de::Visitor<'de> for HeaderVisitor {
            /// Header retaining every uniquely named tensor and metadata entry.
            type Value = RawHeader;

            /// Describes the expected top-level object for Serde's parse diagnostics.
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a SafeTensors header object")
            }

            /// Separates metadata from tensors while rejecting repeated names before insertion.
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

/// String map whose deserializer rejects duplicate keys instead of replacing values.
#[derive(Debug, Default)]
struct UniqueStringMap(
    /// Uniquely named string values in deterministic key order.
    BTreeMap<String, String>,
);

impl<'de> Deserialize<'de> for UniqueStringMap {
    /// Uses the shared duplicate-rejecting map parser for SafeTensors metadata.
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_unique_map(deserializer).map(Self)
    }
}

/// Deserializes string-keyed artifact maps without last-entry-wins ambiguity.
///
/// Serde's ordinary `BTreeMap` implementation replaces duplicate keys. Artifact
/// identities instead require an injective mapping from decoded names to values:
/// even identical repeated entries, including JSON escape-equivalent names, are
/// rejected before decoding the second value. The same parser covers checkpoint
/// weight maps, SafeTensors metadata, and Burnpack tensor/provenance inventories.
pub(super) fn deserialize_unique_map<'de, D, V>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    /// Carries only the value type; the visitor never allocates from a size hint.
    struct UniqueMapVisitor<V>(
        /// Carries the deserialized value type without owning a value.
        std::marker::PhantomData<fn() -> V>,
    );

    impl<'de, V: Deserialize<'de>> de::Visitor<'de> for UniqueMapVisitor<V> {
        /// Decoded values keyed by their unique string names.
        type Value = BTreeMap<String, V>;

        /// Describes the uniqueness requirement in contextual Serde errors.
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("an object with unique string keys")
        }

        /// Rejects repeated decoded keys before they can replace an earlier value.
        fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: de::MapAccess<'de>,
        {
            let mut values = BTreeMap::new();
            while let Some(key) = map.next_key::<String>()? {
                match values.entry(key) {
                    std::collections::btree_map::Entry::Occupied(entry) => {
                        return Err(de::Error::custom(format!(
                            "duplicate map key `{}`",
                            entry.key()
                        )));
                    }
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(map.next_value()?);
                    }
                }
            }
            Ok(values)
        }
    }

    deserializer.deserialize_map(UniqueMapVisitor(std::marker::PhantomData))
}

/// Index fields before shard paths, tensor membership, and byte totals are validated.
#[derive(Debug, Deserialize)]
struct RawIndex {
    /// Optional byte-accounting metadata supplied by the checkpoint producer.
    #[serde(default)]
    metadata: Option<RawIndexMetadata>,
    /// Unique tensor names mapped to shard names relative to the index directory.
    #[serde(deserialize_with = "deserialize_unique_string_map")]
    weight_map: BTreeMap<String, String>,
}

/// Parsed index plus the exact bounded bytes from which it was decoded.
struct RawIndexDocument {
    /// Strictly decoded JSON fields.
    index: RawIndex,
    /// Exact serialized byte length.
    bytes: u64,
    /// SHA-256 over the exact serialized representation.
    sha256: String,
}

/// Byte accounting declared in a checkpoint index.
#[derive(Debug, Deserialize)]
struct RawIndexMetadata {
    /// Nonnegative integral payload byte count, if the index declares one.
    #[serde(default, deserialize_with = "deserialize_optional_integral_u64")]
    total_size: Option<u64>,
}

/// Deserializes an optional non-negative integer from an integer or exact
/// integer-valued JSON decimal.
fn deserialize_optional_integral_u64<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = Option::<Box<serde_json::value::RawValue>>::deserialize(deserializer)?;
    raw.map(|raw| parse_integral_u64_lexeme(raw.get()).map_err(de::Error::custom))
        .transpose()
}

/// Parses one exact JSON-number lexeme without passing through floating point.
///
/// Canonical unsigned integers are accepted across the complete `u64` range.
/// The sole compatibility extension is a decimal point followed exclusively
/// by zeroes, which covers official indexes that serialize byte counts as
/// `55562855904.0`. Fractions, signs, exponents, strings, and overflowing
/// integers fail closed rather than being rounded by IEEE-754 conversion.
fn parse_integral_u64_lexeme(lexeme: &str) -> std::result::Result<u64, String> {
    let (integer, fraction) = match lexeme.split_once('.') {
        Some((integer, fraction)) => (integer, Some(fraction)),
        None => (lexeme, None),
    };
    let canonical_digits = !integer.is_empty() && integer.bytes().all(|byte| byte.is_ascii_digit());
    let exact_zero_fraction = fraction
        .map(|fraction| !fraction.is_empty() && fraction.bytes().all(|byte| byte == b'0'))
        .unwrap_or(true);
    if !canonical_digits || !exact_zero_fraction {
        return Err(format!(
            "metadata.total_size must be an unsigned integer or have only a zero fractional suffix, found {lexeme}"
        ));
    }
    integer.parse::<u64>().map_err(|error| {
        format!("metadata.total_size `{lexeme}` is outside the unsigned 64-bit range: {error}")
    })
}

/// Deserializes a JSON object while rejecting duplicate tensor-name entries.
fn deserialize_unique_string_map<'de, D>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, String>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(UniqueStringMap::deserialize(deserializer)?.0)
}

/// Reads and hashes a size-bounded index, retaining its exact byte identity.
///
/// The reader allows at most one byte beyond the configured bound to detect a file
/// that grows after its initial metadata check; such growth is rejected.
fn read_index(path: &Path, options: &LoadOptions) -> Result<RawIndexDocument> {
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
    let index = serde_json::from_slice(&bytes)
        .map_err(|error| classify_json_error("checkpoint index", path, error))?;
    Ok(RawIndexDocument {
        index,
        bytes: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    })
}

/// Preserves duplicate-key diagnostics separately from other malformed JSON errors.
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
