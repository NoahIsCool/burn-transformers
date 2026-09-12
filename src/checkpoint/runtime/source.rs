//! Audited, bounded sources for pass-two tensor payloads.
//!
//! SafeTensors stores a little-endian, row-major byte interval for each tensor.
//! Pass one has already proved that those intervals are non-overlapping and fit in
//! their shard. This module rechecks the plan against that immutable inventory and
//! seeks directly to each requested interval. The default local source never maps
//! or buffers a complete shard: it owns one file handle and returns one payload.
//!
//! Checkpoint directories are an untrusted boundary. Shard paths are canonicalized
//! again at open time to prevent a replaced symlink from escaping the repository
//! after discovery. File and header lengths are also compared with pass one. A
//! same-size in-place weight mutation is intentionally treated like any other
//! local file mutation; provenance hashes, recorded by the caller's acceptance
//! manifest, are the mechanism for cryptographic identity.

use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
};

#[cfg(unix)]
use std::os::unix::fs::MetadataExt;

use super::super::{
    CheckpointError, CheckpointManifest, PlannedSource, Result, TensorDType, TensorInventory,
};

/// Resource and I/O counters reported by a [`TensorSource`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceStats {
    /// Number of times a shard file was opened.
    pub shard_opens: u64,
    /// Number of tensor payloads returned successfully.
    pub tensor_reads: u64,
    /// Sum of payload bytes returned successfully.
    pub payload_bytes_read: u64,
    /// Maximum number of simultaneously owned shard handles.
    pub max_open_shards: usize,
    /// Shard-open sequence, including a shard reopened after switching away.
    pub shard_open_order: Vec<String>,
}

/// One validated SafeTensors payload returned by a [`TensorSource`].
///
/// Bytes retain their SafeTensors wire representation: row-major and
/// little-endian. The pass-two transformer converts them to Burn's native scalar
/// representation immediately before delivery to a [`super::TensorSink`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTensor {
    /// Physical checkpoint tensor name retained in conversion diagnostics.
    name: String,
    /// Validated logical dimensions in row-major axis order.
    shape: Vec<u64>,
    /// Wire scalar representation used to interpret the payload.
    dtype: TensorDType,
    /// Owned little-endian payload whose length matches the shape and dtype.
    bytes: Vec<u8>,
}

impl SourceTensor {
    /// Creates a source payload after checking shape arithmetic and byte length.
    pub fn new(
        name: impl Into<String>,
        shape: impl Into<Vec<u64>>,
        dtype: TensorDType,
        bytes: Vec<u8>,
    ) -> Result<Self> {
        let name = name.into();
        let shape = shape.into();
        let elements = checked_elements(&shape, &name)?;
        let expected = dtype.storage_bytes(elements)?;
        let actual =
            u64::try_from(bytes.len()).map_err(|_| CheckpointError::InvalidTensorPayload {
                tensor: name.clone(),
                reason: "payload length does not fit in u64".into(),
            })?;
        if actual != expected {
            return Err(CheckpointError::InvalidTensorPayload {
                tensor: name,
                reason: format!(
                    "payload has {actual} bytes, but shape {shape:?} and {dtype} require {expected}"
                ),
            });
        }
        Ok(Self {
            name,
            shape,
            dtype,
            bytes,
        })
    }

    /// Returns the checkpoint key represented by this payload.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the logical row-major dimensions.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns the SafeTensors scalar dtype.
    #[must_use]
    pub const fn dtype(&self) -> TensorDType {
        self.dtype
    }

    /// Returns the little-endian SafeTensors payload bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the payload length in bytes.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    /// Splits the payload into the parts consumed by transformation code.
    pub(super) fn into_parts(self) -> (String, Vec<u64>, TensorDType, Vec<u8>) {
        (self.name, self.shape, self.dtype, self.bytes)
    }
}

/// Supplies individual tensor payloads to the pass-two executor.
///
/// Implementations may read local files, a content-addressed cache, or synthetic
/// test data. They must not reinterpret bytes: returned values use SafeTensors'
/// row-major little-endian representation. The executor independently checks the
/// returned name, shape, dtype, and byte count against each [`PlannedSource`].
pub trait TensorSource {
    /// Validates one descriptor without acquiring or returning its payload.
    ///
    /// Sources backed by a cryptographically authenticated inventory may override
    /// this hook so an executor can bypass source payload I/O for an externally
    /// supplied target. `true` is a strong certificate: it promises that the
    /// descriptor belongs to the exact checkpoint content to which the override is
    /// bound, not merely that it resembles an in-memory manifest entry. The
    /// default `false` retains the ordinary read-and-transform path.
    ///
    /// Implementations must fail closed when they cannot prove byte-level
    /// provenance. A parsed header, canonical path, file length, modification time,
    /// or repository label is not sufficient because each can survive or race a
    /// payload replacement. The built-in local source therefore validates public
    /// descriptors but deliberately returns `false` until checkpoint and sidecar
    /// digests are part of the persisted artifact contract.
    fn validate_tensor_descriptor(&self, _source: &PlannedSource) -> Result<bool> {
        Ok(false)
    }

    /// Reads exactly one planned source tensor.
    fn read_tensor(&mut self, source: &PlannedSource) -> Result<SourceTensor>;

    /// Releases any currently open shard or transport resource.
    ///
    /// The executor calls this before applying aliases and on all normal exits.
    /// It also makes a best-effort call after failures.
    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// Returns current audit counters.
    fn stats(&self) -> SourceStats {
        SourceStats::default()
    }
}

/// Local SafeTensors source backed by a validated [`CheckpointManifest`].
///
/// Repeated reads from the same shard reuse one file handle. Switching shards drops
/// that handle before opening the next, making `max_open_shards <= 1` a structural
/// property rather than a convention.
pub struct LocalShardSource<'a> {
    /// Immutable discovery descriptors used to revalidate execution requests.
    manifest: &'a CheckpointManifest,
    /// Sole live shard handle, replaced when a read switches shards.
    open: Option<OpenShard>,
    /// Cumulative read and open-handle accounting.
    stats: SourceStats,
}

impl<'a> LocalShardSource<'a> {
    /// Creates a lazy source without opening a shard.
    #[must_use]
    pub fn new(manifest: &'a CheckpointManifest) -> Self {
        Self {
            manifest,
            open: None,
            stats: SourceStats::default(),
        }
    }

    /// Returns the pass-one manifest against which reads are audited.
    #[must_use]
    pub const fn manifest(&self) -> &'a CheckpointManifest {
        self.manifest
    }

    /// Opens `relative_path`, first dropping any different open shard.
    fn ensure_open(&mut self, relative_path: &str) -> Result<()> {
        if self
            .open
            .as_ref()
            .is_some_and(|open| open.relative_path == relative_path)
        {
            return Ok(());
        }

        // Drop before opening. There is no execution path with two owned handles.
        self.open = None;
        let shard = self.manifest.shard(relative_path).ok_or_else(|| {
            CheckpointError::PlanManifestMismatch {
                tensor: relative_path.to_owned(),
                reason: "plan references a shard absent from the manifest".into(),
            }
        })?;
        let canonical =
            fs::canonicalize(&shard.path).map_err(|error| CheckpointError::ArtifactChanged {
                path: shard.path.clone(),
                reason: format!("cannot canonicalize shard: {error}"),
            })?;
        if canonical != shard.path || !canonical.starts_with(&self.manifest.root) {
            return Err(CheckpointError::ArtifactChanged {
                path: shard.path.clone(),
                reason: format!("canonical path is now `{}`", canonical.display()),
            });
        }

        let path_metadata =
            fs::symlink_metadata(&canonical).map_err(|error| CheckpointError::ArtifactChanged {
                path: canonical.clone(),
                reason: format!("cannot inspect shard path without following links: {error}"),
            })?;
        if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
            return Err(CheckpointError::ArtifactChanged {
                path: canonical.clone(),
                reason: "shard path is no longer a regular non-symlink file".into(),
            });
        }
        let mut file = File::open(&canonical)
            .map_err(|error| CheckpointError::io("open checkpoint shard", &canonical, error))?;
        let opened_metadata = file
            .metadata()
            .map_err(|error| CheckpointError::io("inspect checkpoint shard", &canonical, error))?;
        if !opened_metadata.is_file() {
            return Err(CheckpointError::ArtifactChanged {
                path: canonical.clone(),
                reason: "opened shard is no longer a regular file".into(),
            });
        }
        #[cfg(unix)]
        if path_metadata.dev() != opened_metadata.dev()
            || path_metadata.ino() != opened_metadata.ino()
        {
            return Err(CheckpointError::ArtifactChanged {
                path: canonical.clone(),
                reason: "shard path changed between no-follow inspection and open".into(),
            });
        }
        let current_bytes = opened_metadata.len();
        if current_bytes != shard.file_bytes {
            return Err(CheckpointError::ArtifactChanged {
                path: canonical,
                reason: format!(
                    "file length is now {current_bytes}, discovered {}",
                    shard.file_bytes
                ),
            });
        }

        let mut header_prefix = [0_u8; 8];
        file.read_exact(&mut header_prefix).map_err(|error| {
            CheckpointError::io("re-read shard header prefix", &canonical, error)
        })?;
        let current_header_bytes = u64::from_le_bytes(header_prefix);
        if current_header_bytes != shard.header_bytes {
            return Err(CheckpointError::ArtifactChanged {
                path: canonical,
                reason: format!(
                    "header length is now {current_header_bytes}, discovered {}",
                    shard.header_bytes
                ),
            });
        }
        let data_start =
            8_u64
                .checked_add(current_header_bytes)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "computing runtime shard data offset",
                })?;
        let described_end = data_start.checked_add(shard.data_bytes).ok_or(
            CheckpointError::ArithmeticOverflow {
                context: "computing runtime shard data end",
            },
        )?;
        if described_end != current_bytes {
            return Err(CheckpointError::ArtifactChanged {
                path: canonical,
                reason: format!(
                    "data section ends at {described_end}, current file ends at {current_bytes}"
                ),
            });
        }

        self.stats.shard_opens =
            self.stats
                .shard_opens
                .checked_add(1)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "counting runtime shard opens",
                })?;
        self.stats.max_open_shards = self.stats.max_open_shards.max(1);
        self.stats.shard_open_order.push(relative_path.to_owned());
        self.open = Some(OpenShard {
            relative_path: relative_path.to_owned(),
            file,
            data_start,
        });
        Ok(())
    }
}

impl TensorSource for LocalShardSource<'_> {
    /// Rechecks a plan descriptor against discovery while withholding content authentication.
    ///
    /// A matching header alone cannot authorize skipping a payload read: its shard may
    /// have been replaced or deleted since discovery.
    fn validate_tensor_descriptor(&self, source: &PlannedSource) -> Result<bool> {
        let observed = self.manifest.tensor(&source.name).ok_or_else(|| {
            CheckpointError::PlanManifestMismatch {
                tensor: source.name.clone(),
                reason: "tensor is absent from the manifest".into(),
            }
        })?;
        validate_descriptor(source, observed)?;
        // The discovery inventory authenticates structure, not payload bytes.
        // Returning `true` here would let a packed sidecar suppress every runtime
        // access to a shard that was deleted or replaced after pass one.
        Ok(false)
    }

    /// Reads one verified tensor interval into owned bytes and records successful I/O.
    ///
    /// The offset is relative to the validated data-section start. Allocation and
    /// byte counters use checked arithmetic; the source retains only one shard handle.
    fn read_tensor(&mut self, source: &PlannedSource) -> Result<SourceTensor> {
        let _ = self.validate_tensor_descriptor(source)?;
        self.ensure_open(&source.shard)?;

        let byte_len = source.data_offsets[1]
            .checked_sub(source.data_offsets[0])
            .ok_or_else(|| CheckpointError::PlanManifestMismatch {
                tensor: source.name.clone(),
                reason: "data offsets are reversed".into(),
            })?;
        let byte_len_usize =
            usize::try_from(byte_len).map_err(|_| CheckpointError::AllocationFailed {
                bytes: byte_len,
                context: "allocating one checkpoint tensor payload",
            })?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len_usize)
            .map_err(|_| CheckpointError::AllocationFailed {
                bytes: byte_len,
                context: "allocating one checkpoint tensor payload",
            })?;
        bytes.resize(byte_len_usize, 0);

        let open = self
            .open
            .as_mut()
            .ok_or_else(|| CheckpointError::PlanManifestMismatch {
                tensor: source.name.clone(),
                reason: "shard source did not retain its newly opened file".into(),
            })?;
        let absolute_start = open.data_start.checked_add(source.data_offsets[0]).ok_or(
            CheckpointError::ArithmeticOverflow {
                context: "computing runtime tensor file offset",
            },
        )?;
        open.file
            .seek(SeekFrom::Start(absolute_start))
            .map_err(|error| {
                CheckpointError::io("seek to checkpoint tensor", &source.shard, error)
            })?;
        open.file
            .read_exact(&mut bytes)
            .map_err(|error| CheckpointError::io("read checkpoint tensor", &source.shard, error))?;

        let payload = SourceTensor::new(
            source.name.clone(),
            source.shape.clone(),
            source.dtype,
            bytes,
        )?;
        self.stats.tensor_reads =
            self.stats
                .tensor_reads
                .checked_add(1)
                .ok_or(CheckpointError::ArithmeticOverflow {
                    context: "counting runtime tensor reads",
                })?;
        self.stats.payload_bytes_read = self.stats.payload_bytes_read.checked_add(byte_len).ok_or(
            CheckpointError::ArithmeticOverflow {
                context: "summing runtime payload bytes",
            },
        )?;
        Ok(payload)
    }

    /// Releases the current shard handle while retaining accumulated audit counters.
    fn close(&mut self) -> Result<()> {
        self.open = None;
        Ok(())
    }

    /// Returns a snapshot of file and payload counters without resetting them.
    fn stats(&self) -> SourceStats {
        self.stats.clone()
    }
}

/// A single open shard and the absolute start of its tensor data section.
struct OpenShard {
    /// Root-relative identity of the currently open shard.
    relative_path: String,
    /// Read-only file handle retained across same-shard reads.
    file: File,
    /// Absolute byte offset immediately after the SafeTensors header.
    data_start: u64,
}

/// Revalidates a mutable public load-plan source against immutable discovery data.
fn validate_descriptor(source: &PlannedSource, observed: &TensorInventory) -> Result<()> {
    let mismatch = if source.name != observed.name {
        Some("source name differs")
    } else if source.shard != observed.shard {
        Some("shard name differs")
    } else if source.shape != observed.shape {
        Some("shape differs")
    } else if source.dtype != observed.dtype {
        Some("dtype differs")
    } else if source.data_offsets != observed.data_offsets {
        Some("data offsets differ")
    } else {
        None
    };
    if let Some(reason) = mismatch {
        return Err(CheckpointError::PlanManifestMismatch {
            tensor: source.name.clone(),
            reason: reason.into(),
        });
    }
    Ok(())
}

/// Computes a scalar count without trusting shape multiplication.
fn checked_elements(shape: &[u64], tensor: &str) -> Result<u64> {
    shape.iter().try_fold(1_u64, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .ok_or_else(|| CheckpointError::InvalidTensorPayload {
                tensor: tensor.to_owned(),
                reason: "shape element count overflows u64".into(),
            })
    })
}

#[cfg(test)]
#[path = "source_boundary_tests.rs"]
mod boundary_tests;
