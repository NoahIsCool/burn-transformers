//! Minimal in-process capture of CubeCL kernel-launch spans.
//!
//! Burn's `tracing` feature instruments the final `ComputeClient::launch_inner`
//! boundary with `kernel.name` and `kernel.id`. Capturing those two fields proves
//! which compiled operation received the actual bindings; observing a QFloat
//! parameter before and after a model call alone cannot distinguish packed
//! execution from a hidden whole-matrix dequantization fallback.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use tracing::{
    Event, Metadata, Subscriber,
    field::{Field, Visit},
    span::{Attributes, Id, Record},
};

/// One kernel submitted through CubeCL's checked launch boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KernelLaunch {
    /// Fully qualified generated-kernel name.
    pub(crate) name: String,
    /// Stable, configuration-bearing CubeCL kernel identifier.
    pub(crate) id: String,
}

/// Thread-safe subscriber state shared with the capture caller.
#[derive(Clone, Default)]
struct KernelLaunchSubscriber {
    /// Quantized or dequantization launches captured in submission order.
    launches: Arc<Mutex<Vec<KernelLaunch>>>,
    /// Shared counter allocating distinct nonzero tracing span identifiers.
    next_span: Arc<AtomicU64>,
}

/// Visitor for the two display-valued fields attached to `launch_inner`.
#[derive(Default)]
struct KernelFieldVisitor {
    /// Generated kernel name, populated only when its tracing field is present.
    name: Option<String>,
    /// Configuration-bearing kernel identifier collected from the launch span.
    id: Option<String>,
}

impl KernelFieldVisitor {
    /// Records a field after stripping `tracing`'s debug-string quotes.
    fn record_text(&mut self, field: &Field, value: String) {
        let value = value
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .unwrap_or(&value)
            .to_owned();
        match field.name() {
            "kernel.name" => self.name = Some(value),
            "kernel.id" => self.id = Some(value),
            _ => {}
        }
    }
}

impl Visit for KernelFieldVisitor {
    /// Collects a debug-formatted kernel field, removing enclosing string quotes.
    fn record_debug(&mut self, field: &Field, value: &dyn core::fmt::Debug) {
        self.record_text(field, format!("{value:?}"));
    }

    /// Collects a string-valued kernel field, removing enclosing quotes if present.
    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_text(field, value.to_owned());
    }
}

impl Subscriber for KernelLaunchSubscriber {
    /// Enables only the CubeCL launch boundary used to identify actual kernels.
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.name() == "launch_inner"
    }

    /// Records packed or dequantization launches and allocates a distinct span ID.
    /// Panics if the shared launch mutex was poisoned or the span counter is exhausted.
    fn new_span(&self, attributes: &Attributes<'_>) -> Id {
        let mut visitor = KernelFieldVisitor::default();
        attributes.record(&mut visitor);
        if let (Some(name), Some(id)) = (visitor.name, visitor.id)
            && (id.contains("QuantizedView") || name.contains("dequantize"))
        {
            self.launches
                .lock()
                .expect("kernel trace mutex must not be poisoned")
                .push(KernelLaunch { name, id });
        }
        Id::from_u64(self.next_span.fetch_add(1, Ordering::Relaxed) + 1)
    }

    /// Ignores later field updates because launch identity is captured at span creation.
    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    /// Ignores causal span links; launch capture does not reconstruct execution graphs.
    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    /// Ignores standalone events because kernel identities are carried by spans.
    fn event(&self, _event: &Event<'_>) {}

    /// Ignores span entry because submission order is recorded when spans are created.
    fn enter(&self, _span: &Id) {}

    /// Ignores span exit because no duration measurements are collected.
    fn exit(&self, _span: &Id) {}
}

/// Executes one closure and returns its quantized/dequantized kernel launches.
///
/// The subscriber is thread-local by design. CubeCL creates each launch span on
/// the calling thread before submitting asynchronous GPU work, so the scope
/// covers both eager execution and Burn fusion flushes triggered by readback.
/// Filtering at span creation avoids retaining the very large IDs of unrelated
/// elementwise kernels during full-model audits.
pub(crate) fn capture_kernel_launches<T>(operation: impl FnOnce() -> T) -> (T, Vec<KernelLaunch>) {
    let subscriber = KernelLaunchSubscriber::default();
    let launches = subscriber.launches.clone();
    let output = tracing::subscriber::with_default(subscriber, operation);
    let launches = Arc::try_unwrap(launches)
        .expect("capture must release its only launch-state owner")
        .into_inner()
        .expect("kernel trace mutex must not be poisoned");
    (output, launches)
}

/// Selects launches whose kernel configuration consumes packed QFloat storage.
pub(crate) fn packed_matmul_launches(trace: &[KernelLaunch]) -> Vec<&KernelLaunch> {
    trace
        .iter()
        .filter(|launch| launch.id.contains("QuantizedView") && launch.id.contains("PackedU32"))
        .collect()
}

/// Removes pretty-printer whitespace so structured ID fragments can be asserted.
pub(crate) fn compact_kernel_id(launch: &KernelLaunch) -> String {
    launch
        .id
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

/// Extracts the generated kernel names observed during explicit dequantization.
///
/// Matching names, rather than full IDs, intentionally catches an F16 fallback
/// after calibrating the public F32 dequantize operation: output dtype changes
/// the ID but not the mathematical kernel family.
pub(crate) fn calibrated_dequantization_kernel_names(trace: &[KernelLaunch]) -> Vec<String> {
    let mut names = trace
        .iter()
        .filter(|launch| launch.name.contains("dequantize"))
        .map(|launch| launch.name.clone())
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

/// Returns launches belonging to a calibrated whole-tensor dequant kernel family.
pub(crate) fn whole_weight_dequantization_launches<'a>(
    trace: &'a [KernelLaunch],
    calibrated_names: &[String],
) -> Vec<&'a KernelLaunch> {
    trace
        .iter()
        .filter(|launch| {
            calibrated_names
                .iter()
                .any(|name| launch.name == *name || launch.name.contains("dequantize"))
        })
        .collect()
}
