//! Test-only fixtures for mod.rs; excluded from non-test compiler inputs.

use std::{cell::RefCell, io};

use super::*;
use crate::checkpoint::{MemoryEstimate, PlannedSource, PlannedTransform};

/// Synthetic source whose descriptor certificates can be granted or rejected.
#[derive(Default)]
struct AuditedSource {
    certified: bool,
    uncertified_name: Option<String>,
    rejected_name: Option<String>,
    validated: RefCell<Vec<String>>,
    reads: u64,
    closes: usize,
}

impl TensorSource for AuditedSource {
    /// Records every certificate request, including requests that fail closed.
    fn validate_tensor_descriptor(
        &self,
        source: &PlannedSource,
    ) -> crate::checkpoint::Result<bool> {
        self.validated.borrow_mut().push(source.name.clone());
        if self.rejected_name.as_deref() == Some(&source.name) {
            return Err(CheckpointError::PlanManifestMismatch {
                tensor: source.name.clone(),
                reason: "synthetic certificate mismatch".into(),
            });
        }
        Ok(self.certified && self.uncertified_name.as_deref() != Some(&source.name))
    }

    /// Returns deterministic row-major values so real transform paths execute.
    fn read_tensor(&mut self, source: &PlannedSource) -> crate::checkpoint::Result<SourceTensor> {
        self.reads += 1;
        let elements = source.shape.iter().product::<u64>();
        let bytes = (1..=elements)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect();
        SourceTensor::new(&source.name, source.shape.clone(), source.dtype, bytes)
    }

    /// Records release even when validation fails before the first read.
    fn close(&mut self) -> crate::checkpoint::Result<()> {
        self.closes += 1;
        Ok(())
    }

    /// Exposes physical read counts to success and partial-failure reports.
    fn stats(&self) -> SourceStats {
        SourceStats {
            tensor_reads: self.reads,
            ..SourceStats::default()
        }
    }
}

/// Minimal destination recording successful writes in their execution order.
#[derive(Default)]
struct RecordingSink(Vec<(String, TensorData)>);

impl TensorSink for RecordingSink {
    type Error = io::Error;

    /// Retains final buffers to distinguish overrides from ordinary payloads.
    fn write_tensor(&mut self, target: &str, data: TensorData) -> io::Result<()> {
        self.0.push((target.into(), data));
        Ok(())
    }

    /// Rejects unexpected aliases, which these isolated override plans omit.
    fn alias_tensor(&mut self, _target: &str, _source_target: &str) -> io::Result<()> {
        Err(io::Error::other("unexpected alias"))
    }
}

/// Builds a compact descriptor with physically ordered, disjoint intervals.
fn descriptor(name: &str, elements: u64, offset: u64) -> PlannedSource {
    PlannedSource {
        name: name.into(),
        shard: "model.safetensors".into(),
        shape: vec![elements],
        dtype: TensorDType::F32,
        data_offsets: [offset, offset + elements * 4],
    }
}

/// Wraps validated operations with an explicit bounded staging allowance.
fn plan(tensors: Vec<PlannedTensor>, peak: u64) -> LoadPlan {
    let plan = LoadPlan {
        equivalences: Vec::new(),
        tensors,
        aliases: Vec::new(),
        report: LoadReport::default(),
        memory: MemoryEstimate {
            peak_host_staging_bytes: peak,
            ..MemoryEstimate::default()
        },
    };
    transform::validate_plan(&plan).unwrap();
    plan
}

/// Splits one fused four-element parameter into two two-element targets.
fn fused_plan() -> LoadPlan {
    let source = descriptor("fused", 4, 0);
    plan(
        ["a", "b"]
            .into_iter()
            .enumerate()
            .map(|(index, target)| PlannedTensor {
                target: target.into(),
                sources: vec![source.clone()],
                transform: PlannedTransform::Slice {
                    axis: 0,
                    start: (index * 2) as u64,
                    length: 2,
                },
                shape: vec![2],
                dtype: TensorDType::F32,
            })
            .collect(),
        24,
    )
}

/// Selects each target once even if a predicate would change on later calls.
#[test]
fn override_selection_is_frozen_and_uncertified_groups_read_once() {
    for certified in [false, true] {
        let mut source = AuditedSource {
            certified,
            ..AuditedSource::default()
        };
        let mut sink = RecordingSink::default();
        let calls = std::cell::Cell::new(0);
        let outcome = execute_load_plan_with_overrides(
            &fused_plan(),
            &mut source,
            &mut sink,
            |_, _| {
                let previous = calls.replace(calls.get() + 1);
                previous < 2
            },
            |_, _| Ok(Some(TensorData::new(vec![9.0_f32; 2], [2]))),
        )
        .unwrap();
        assert_eq!(calls.get(), 2);
        assert_eq!(source.reads, u64::from(!certified));
        assert_eq!(source.validated.borrow().as_slice(), ["fused", "fused"]);
        assert_eq!(source.closes, 1);
        assert_eq!(outcome.report.applied, ["a", "b"]);
        assert_eq!(outcome.stats.source_reads, source.reads);
        assert_eq!(outcome.stats.source.tensor_reads, source.reads);
        assert_eq!(
            outcome.stats.peak_host_staging_bytes,
            if certified { 8 } else { 24 }
        );
        for (_, data) in sink.0 {
            assert_eq!(data.to_vec::<f32>().unwrap(), vec![9.0; 2]);
        }
    }
}

/// Keeps a shared fused payload alive only when an ordinary sibling needs it.
#[test]
fn partial_override_retains_one_shared_source_and_exact_sibling_values() {
    let mut source = AuditedSource {
        certified: true,
        ..AuditedSource::default()
    };
    let mut sink = RecordingSink::default();
    let outcome = execute_load_plan_with_overrides(
        &fused_plan(),
        &mut source,
        &mut sink,
        |_, tensor| tensor.target == "a",
        |_, _| Ok(Some(TensorData::new(vec![9.0_f32; 2], [2]))),
    )
    .unwrap();
    assert_eq!(source.reads, 1);
    assert_eq!(source.validated.borrow().as_slice(), ["fused"]);
    assert_eq!(outcome.stats.peak_host_staging_bytes, 24);
    assert_eq!(outcome.stats.source_payload_bytes, 16);
    assert_eq!(sink.0[0].1.to_vec::<f32>().unwrap(), vec![9.0; 2]);
    assert_eq!(sink.0[1].1.to_vec::<f32>().unwrap(), vec![3.0, 4.0]);
}

/// Requires every descriptor of a concatenation to authenticate the override.
#[test]
fn one_uncertified_concat_source_requires_complete_fallback_validation() {
    let plan = plan(
        vec![PlannedTensor {
            target: "combined".into(),
            sources: vec![descriptor("first", 1, 0), descriptor("second", 1, 4)],
            transform: PlannedTransform::Concatenate { axis: 0 },
            shape: vec![2],
            dtype: TensorDType::F32,
        }],
        12,
    );
    for rejected in [false, true] {
        let mut source = AuditedSource {
            certified: true,
            uncertified_name: Some("second".into()),
            rejected_name: rejected.then(|| "second".into()),
            ..AuditedSource::default()
        };
        let mut sink = RecordingSink::default();
        let providers = std::cell::Cell::new(0);
        let result = execute_load_plan_with_overrides(
            &plan,
            &mut source,
            &mut sink,
            |_, _| true,
            |_, _| {
                providers.set(providers.get() + 1);
                Ok(Some(TensorData::new(vec![9.0_f32; 2], [2])))
            },
        );
        assert_eq!(source.validated.borrow().as_slice(), ["first", "second"]);
        assert_eq!(source.closes, 1);
        if rejected {
            let error = result.unwrap_err();
            assert_eq!(error.failed_target.as_deref(), Some("combined"));
            assert!(matches!(error.cause, ExecutionFailureCause::Checkpoint(
                CheckpointError::PlanManifestMismatch { tensor, .. }) if tensor == "second"));
            assert!(error.report.applied.is_empty());
            assert!(sink.0.is_empty());
            assert_eq!(source.reads, 0);
            assert_eq!(providers.get(), 0);
        } else {
            let outcome = result.unwrap();
            assert_eq!(source.reads, 2);
            assert_eq!(providers.get(), 1);
            assert_eq!(outcome.stats.source_payload_bytes, 8);
            assert_eq!(sink.0[0].1.to_vec::<f32>().unwrap(), vec![9.0; 2]);
        }
    }
}

/// Returns exact partial progress and an inspectable error chain on overrides.
#[test]
fn failed_override_provider_preserves_prior_writes_and_closes_source() {
    for missing in [false, true] {
        let mut source = AuditedSource {
            certified: true,
            ..AuditedSource::default()
        };
        let mut sink = RecordingSink::default();
        let error = execute_load_plan_with_overrides(
            &fused_plan(),
            &mut source,
            &mut sink,
            |_, _| true,
            |_, tensor| {
                if tensor.target == "a" {
                    Ok(Some(TensorData::new(vec![9.0_f32; 2], [2])))
                } else if missing {
                    Ok(None)
                } else {
                    Err(io::Error::other("packed tensor unavailable"))
                }
            },
        )
        .unwrap_err();
        assert_eq!(error.failed_target.as_deref(), Some("b"));
        assert_eq!(error.report.applied, ["a"]);
        assert_eq!(error.stats.tensors_applied, 1);
        assert_eq!(error.stats.delivered_tensor_bytes, 8);
        assert_eq!(source.closes, 1);
        assert_eq!(source.reads, 0);
        assert_eq!(sink.0.len(), 1);
        assert!(error.to_string().contains("failed at `b`"));
        let cause = Error::source(error.as_ref()).unwrap();
        let underlying = cause.source().unwrap();
        if missing {
            assert!(underlying.downcast_ref::<CheckpointError>().is_some());
            assert!(matches!(error.cause, ExecutionFailureCause::Checkpoint(
                CheckpointError::InvalidMapping { message }) if message.contains("returned no tensor")));
        } else {
            assert!(cause.to_string().contains("tensor sink failed"));
            assert_eq!(
                underlying.downcast_ref::<io::Error>().unwrap().kind(),
                io::ErrorKind::Other
            );
            assert!(matches!(error.cause, ExecutionFailureCause::Sink(_)));
        }
    }
}

/// Rejects a logically wrong override shape before handing any buffer to Burn.
#[test]
fn override_shape_mismatch_is_contextual_and_never_installed() {
    let mut source = AuditedSource {
        certified: true,
        ..AuditedSource::default()
    };
    let mut sink = RecordingSink::default();
    let error = execute_load_plan_with_overrides(
        &fused_plan(),
        &mut source,
        &mut sink,
        |_, _| true,
        |_, _| Ok(Some(TensorData::new(vec![1.0_f32; 2], [1, 2]))),
    )
    .unwrap_err();
    assert!(matches!(error.cause, ExecutionFailureCause::Checkpoint(
        CheckpointError::InvalidTensorPayload { tensor, reason })
        if tensor == "a" && reason.contains("override shape")));
    assert!(sink.0.is_empty());
    assert!(error.report.applied.is_empty());
    assert_eq!(source.closes, 1);
    assert_eq!(source.reads, 0);
}

/// Includes a retained fused payload when rejecting an oversized replacement.
#[test]
fn override_staging_limit_accounts_for_shared_payload_overlap() {
    let mut source = AuditedSource {
        certified: true,
        ..AuditedSource::default()
    };
    let mut sink = RecordingSink::default();
    let error = execute_load_plan_with_overrides(
        &fused_plan(),
        &mut source,
        &mut sink,
        |_, tensor| tensor.target == "a",
        |_, _| Ok(Some(TensorData::new(vec![1.0_f64; 2], [2]))),
    )
    .unwrap_err();
    assert!(matches!(error.cause, ExecutionFailureCause::Checkpoint(
        CheckpointError::MemoryEstimateExceeded { tensor, required: 32, planned: 24 }) if tensor == "a"));
    assert_eq!(error.stats.peak_host_staging_bytes, 32);
    assert_eq!(error.stats.largest_output_tensor_bytes, 16);
    assert_eq!(source.reads, 1);
    assert_eq!(source.closes, 1);
    assert!(sink.0.is_empty());
}

/// Verifies accounting overflow cannot leave an unreported destination write.
#[test]
fn install_accounting_overflow_is_atomic_before_sink_mutation() {
    let tensor = &fused_plan().tensors[0];
    for count_overflow in [false, true] {
        let mut stats = ExecutionStats {
            tensors_applied: if count_overflow { usize::MAX } else { 3 },
            delivered_tensor_bytes: if count_overflow { 24 } else { u64::MAX - 7 },
            ..ExecutionStats::default()
        };
        let before = stats.clone();
        let mut report = LoadReport::default();
        report.record_applied("previous");
        let mut sink = RecordingSink::default();
        let result = install_tensor(
            tensor,
            TensorData::new(vec![1.0_f32; 2], [2]),
            &mut sink,
            &mut report,
            &mut stats,
        );
        assert!(
            matches!(result, Err(InstallError::Checkpoint(CheckpointError::ArithmeticOverflow { context }))
            if context == if count_overflow { "counting applied target tensors" } else { "summing target bytes delivered to the sink" })
        );
        assert_eq!(stats, before);
        assert_eq!(report.applied, ["previous"]);
        assert!(sink.0.is_empty());
    }
}

/// Keeps plan-wide error messages distinct from target-specific write failures.
#[test]
fn invalid_plan_has_no_target_and_preserves_checkpoint_error_chain() {
    let mut invalid = fused_plan();
    invalid.tensors[0].target.clear();
    let mut source = AuditedSource::default();
    let mut sink = RecordingSink::default();
    let error = execute_load_plan(&invalid, &mut source, &mut sink).unwrap_err();
    assert!(error.failed_target.is_none());
    assert!(
        error
            .to_string()
            .starts_with("checkpoint execution failed: ")
    );
    assert!(
        Error::source(error.as_ref())
            .unwrap()
            .source()
            .unwrap()
            .downcast_ref::<CheckpointError>()
            .is_some()
    );
    assert!(sink.0.is_empty());
    assert_eq!(source.reads, 0);
    assert_eq!(source.closes, 1);
}
