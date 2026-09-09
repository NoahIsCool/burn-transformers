//! Pure validation and reporting mathematics for reproducible inference benchmarks.
//!
//! Samples are wall-clock seconds measured between completed backend synchronization points.
//! Throughput divides valid work by elapsed time; padding and EOS-shortened decode loops must
//! never inflate the numerator. Quantiles use the nearest-rank definition for p95 and the mean
//! of the two central order statistics for an even-size median.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Minimum complete untimed executions of every measured shape.
pub const WARMUP_REPETITIONS: usize = 3;
/// Number of measured repetitions required by `BENCHMARKS.md`.
pub const MEASURED_REPETITIONS: usize = 10;
/// Fixed number of valid cached decode steps, independent of EOS.
pub const DECODE_STEPS: usize = 128;

/// Raw and aggregate wall-time evidence for one homogeneous phase.
#[derive(Debug, Clone, Serialize)]
pub struct Distribution {
    /// Original samples, in execution order, measured in seconds.
    pub samples_seconds: Vec<f64>,
    /// Median elapsed time.
    pub median_seconds: f64,
    /// Nearest-rank 95th percentile elapsed time.
    pub p95_seconds: f64,
    /// Smallest observed elapsed time.
    pub minimum_seconds: f64,
    /// Largest observed elapsed time.
    pub maximum_seconds: f64,
    /// Valid tokens processed by each sample, or absent for a latency-only phase.
    pub valid_tokens_per_sample: Option<usize>,
    /// Per-sample throughput, preserving the pairing with original elapsed times.
    pub tokens_per_second: Option<Vec<f64>>,
}

/// Validates raw timing samples and computes reproducible order statistics.
pub fn summarize(
    samples: &[f64],
    valid_tokens: Option<usize>,
    minimum: usize,
) -> Result<Distribution, String> {
    if minimum == 0 || samples.len() < minimum {
        return Err("not enough measured repetitions".into());
    }
    if samples
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("timing samples must be finite and strictly positive".into());
    }
    if valid_tokens == Some(0) {
        return Err("throughput requires nonzero valid work".into());
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    let median = if ordered.len().is_multiple_of(2) {
        ordered[middle - 1] / 2.0 + ordered[middle] / 2.0
    } else {
        ordered[middle]
    };
    let rank = ordered.len() - ordered.len() / 20;
    let rates = valid_tokens.map(|tokens| {
        samples
            .iter()
            .map(|&seconds| tokens as f64 / seconds)
            .collect::<Vec<_>>()
    });
    if rates
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
    {
        return Err("throughput is not representable as a finite number".into());
    }
    Ok(Distribution {
        samples_seconds: samples.to_vec(),
        median_seconds: median,
        p95_seconds: ordered[rank - 1],
        minimum_seconds: ordered[0],
        maximum_seconds: ordered[ordered.len() - 1],
        valid_tokens_per_sample: valid_tokens,
        tokens_per_second: rates,
    })
}

/// Produces the canonical output identity required by the benchmark protocol.
pub fn token_hash(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Analytical, distinct T5 self-cache and immutable cross-cache payloads.
#[derive(Debug, Clone, Serialize)]
pub struct T5CacheAccounting {
    /// Self-attention KV bytes for one additional target position in one sequence.
    pub self_kv_bytes_per_token_per_sequence: u64,
    /// Preallocated decoder self-attention key and value payload.
    pub self_kv_allocated_bytes: u64,
    /// Immutable cross-attention key/value payload for the bound encoder output.
    pub cross_kv_bytes: u64,
    /// Retained encoder residual output, counted separately from cross K/V.
    pub retained_encoder_hidden_bytes: u64,
    /// Analytical sum of the above payloads; excludes allocator and Rust metadata overhead.
    pub total_tensor_payload_bytes: u64,
}

/// Computes cache payloads with checked arithmetic and no device allocation.
#[allow(
    clippy::too_many_arguments,
    reason = "the independent analytical factors are recorded individually"
)]
pub fn t5_cache_accounting(
    layers: u64,
    heads: u64,
    head_dim: u64,
    hidden: u64,
    scalar_bytes: u64,
    batch: u64,
    target_capacity: u64,
    source_length: u64,
) -> Result<T5CacheAccounting, String> {
    let product = |values: &[u64]| -> Result<u64, String> {
        if values.contains(&0) {
            return Err("cache dimensions must be nonzero".into());
        }
        values.iter().try_fold(1_u64, |total, value| {
            total
                .checked_mul(*value)
                .ok_or_else(|| "cache byte arithmetic overflow".into())
        })
    };
    let per_token = product(&[layers, 2, heads, head_dim, scalar_bytes])?;
    let self_kv = product(&[per_token, batch, target_capacity])?;
    let cross_kv = product(&[per_token, batch, source_length])?;
    let encoder = product(&[batch, source_length, hidden, scalar_bytes])?;
    let total = self_kv
        .checked_add(cross_kv)
        .and_then(|value| value.checked_add(encoder))
        .ok_or("cache sum overflow")?;
    Ok(T5CacheAccounting {
        self_kv_bytes_per_token_per_sequence: per_token,
        self_kv_allocated_bytes: self_kv,
        cross_kv_bytes: cross_kv,
        retained_encoder_hidden_bytes: encoder,
        total_tensor_payload_bytes: total,
    })
}

/// Operator-supplied environment facts that cannot be inferred reliably from a backend name.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunEnvironment {
    /// Physical machine model.
    pub machine_model: String,
    /// CPU model used for preprocessing and host inference.
    pub cpu: String,
    /// Accelerator model, or `none` for the CPU-only series.
    pub accelerator: String,
    /// Physical/unified memory capacity, not an allocation measurement.
    pub physical_memory_bytes: u64,
    /// Exact operating system version.
    pub operating_system: String,
    /// Driver/runtime version, including OS-provided Metal version where applicable.
    pub driver_runtime: String,
    /// Power profile active during the run.
    pub power_mode: String,
    /// Whether the operator reserved the accelerator for this benchmark.
    pub accelerator_exclusive: bool,
    /// Description of concurrent work, or `none` for an idle host.
    pub other_workloads: String,
    /// Exact command that invoked the benchmark executable or Cargo test.
    pub command: String,
}

impl RunEnvironment {
    /// Rejects incomplete environment records before expensive inference.
    pub fn validate(&self) -> Result<(), String> {
        if [
            &self.machine_model,
            &self.cpu,
            &self.accelerator,
            &self.operating_system,
            &self.driver_runtime,
            &self.power_mode,
            &self.other_workloads,
            &self.command,
        ]
        .iter()
        .any(|field| field.trim().is_empty())
            || self.physical_memory_bytes == 0
        {
            return Err("benchmark environment fields must be explicit and nonempty".into());
        }
        Ok(())
    }
}

/// Returns whether this is a separate, timing-incomparable process-memory run.
///
/// The private Unix socket is test-harness instrumentation only. Caching the
/// optional endpoint means normal benchmark iterations never probe the environment
/// or perform IPC. No network behavior is added to the runtime library.
pub fn memory_profiling_enabled() -> bool {
    memory_endpoint().is_some()
}

/// Resolves the opt-in supervisor endpoint once for this benchmark process.
fn memory_endpoint() -> &'static Option<std::path::PathBuf> {
    static ENDPOINT: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    ENDPOINT.get_or_init(|| std::env::var_os("TRANSFORMERS_BENCH_MEMORY_SOCKET").map(Into::into))
}

/// One synchronized memory phase; callers must finish it after backend completion.
///
/// Starting a phase first blocks while the supervisor samples an idle baseline.
/// The begin/end acknowledgements conservatively bracket the completed operation.
/// This changes scheduling even when outside a timer, so instrumented timing
/// samples must never be compared with the normal benchmark series.
#[must_use = "finish the memory phase after the backend synchronization boundary"]
pub struct MemoryPhase {
    /// Serial protocol identity, absent for completely uninstrumented execution.
    id: Option<u64>,
}

impl MemoryPhase {
    /// Samples idle memory before releasing the child to execute the named phase.
    pub fn begin(label: &str) -> Self {
        if !memory_profiling_enabled() {
            return Self { id: None };
        }
        #[cfg(target_os = "macos")]
        {
            let mut channel = memory_channel().lock().expect("memory channel mutex");
            Self {
                id: Some(channel.begin(label).expect("memory supervisor phase begin")),
            }
        }
        #[cfg(not(target_os = "macos"))]
        panic!("process-memory phase instrumentation is macOS-only: {label}");
    }

    /// Records the completed phase after the caller has synchronized its backend.
    pub fn finish(self) {
        if let Some(id) = self.id {
            #[cfg(target_os = "macos")]
            memory_channel()
                .lock()
                .expect("memory channel mutex")
                .finish(id)
                .expect("memory supervisor phase end");
            #[cfg(not(target_os = "macos"))]
            panic!("unexpected memory phase {id} on a non-macOS host");
        }
    }
}

/// Connects once to the supervisor's private socket; stale endpoints fail closed.
#[cfg(target_os = "macos")]
fn memory_channel() -> &'static std::sync::Mutex<MemoryChannel> {
    static CHANNEL: std::sync::OnceLock<std::sync::Mutex<MemoryChannel>> =
        std::sync::OnceLock::new();
    CHANNEL.get_or_init(|| {
        std::sync::Mutex::new(
            MemoryChannel::connect(memory_endpoint().as_ref().unwrap())
                .expect("connect memory supervisor"),
        )
    })
}

/// Serial, bounded newline-JSON protocol used only by opt-in benchmark tests.
#[cfg(target_os = "macos")]
struct MemoryChannel {
    /// One locally connected stream; no listener is created by the child.
    stream: std::os::unix::net::UnixStream,
    /// Monotonically increasing identity prevents accidental cross-phase ACK reuse.
    next_id: u64,
    /// Nested phases are rejected so each baseline belongs to precisely one phase.
    active: Option<u64>,
}

#[cfg(target_os = "macos")]
impl MemoryChannel {
    /// Applies bounded reads/writes so a failed supervisor cannot hang inference.
    fn connect(path: &std::path::Path) -> std::io::Result<Self> {
        let stream = std::os::unix::net::UnixStream::connect(path)?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(std::time::Duration::from_secs(5)))?;
        Ok(Self {
            stream,
            next_id: 0,
            active: None,
        })
    }

    /// Sends one bounded event and requires an exact matching positive ACK.
    fn request(&mut self, id: u64, event: &str, label: &str) -> std::io::Result<()> {
        use std::io::{Read, Write};
        let request = serde_json::json!({"schema_version":1,"id":id,"event":event,"phase":label});
        let mut encoded = serde_json::to_vec(&request)?;
        encoded.push(b'\n');
        let timeout = self
            .stream
            .read_timeout()?
            .unwrap_or(std::time::Duration::from_secs(5));
        // Nonblocking reads enforce one total deadline, including fragmented
        // replies. Mutating Darwin SO_RCVTIMEO after a peer closes can reject a
        // perfectly buffered final ACK; no socket timeout is changed mid-reply.
        self.stream.set_nonblocking(true)?;
        let started = std::time::Instant::now();
        self.stream.write_all(&encoded)?;
        let mut response = Vec::new();
        loop {
            if started.elapsed() >= timeout {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "memory ACK deadline exceeded",
                ));
            }
            let mut buffer = [0_u8; 4097];
            let count = match self.stream.read(&mut buffer) {
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "memory supervisor closed",
                    ));
                }
                Ok(count) => count,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                Err(error) => return Err(error),
            };
            response.extend_from_slice(&buffer[..count]);
            if response.len() > 4096 {
                return Err(std::io::Error::other("oversized memory ACK"));
            }
            if let Some(newline) = response.iter().position(|&byte| byte == b'\n') {
                if newline + 1 != response.len() {
                    return Err(std::io::Error::other("pipelined memory ACK"));
                }
                response.pop();
                break;
            }
        }
        let ack: serde_json::Value = serde_json::from_slice(&response)?;
        if ack != serde_json::json!({"ok":true,"id":id,"event":event}) {
            return Err(std::io::Error::other("invalid memory supervisor ACK"));
        }
        Ok(())
    }

    /// Enforces non-nested phase ordering and samples idle before the begin boundary.
    fn begin(&mut self, label: &str) -> std::io::Result<u64> {
        if self.active.is_some()
            || label.is_empty()
            || label.len() > 128
            || !label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
        {
            return Err(std::io::Error::other("invalid or nested memory phase"));
        }
        let id = self.next_id;
        self.next_id = id
            .checked_add(1)
            .ok_or_else(|| std::io::Error::other("memory phase ID overflow"))?;
        self.request(id, "idle", label)?;
        self.request(id, "begin", label)?;
        self.active = Some(id);
        Ok(id)
    }

    /// Ends only the currently active phase; the supervisor resolves its sampled maximum.
    fn finish(&mut self, id: u64) -> std::io::Result<()> {
        if self.active != Some(id) {
            return Err(std::io::Error::other("memory phase ID mismatch"));
        }
        self.request(id, "end", "")?;
        self.active = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Disabled instrumentation has no external side effects or required supervisor.
    #[test]
    fn disabled_memory_phase_is_a_noop() {
        MemoryPhase { id: None }.finish();
    }

    /// Exercises the actual supervisor handshake and a touched CPU allocation, never a GPU.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "CPU-only IPC smoke: run this exact test under scripts/profile_process_memory.py"]
    fn memory_marker_supervised_cpu_smoke() {
        assert!(
            memory_profiling_enabled(),
            "start the private memory supervisor first"
        );
        let phase = MemoryPhase::begin("synthetic_cpu_allocation");
        let mut allocation = vec![0_u8; 8 * 1024 * 1024];
        for page in allocation.chunks_mut(4096) {
            page[0] = 1;
        }
        std::hint::black_box(&allocation);
        std::thread::sleep(std::time::Duration::from_millis(20));
        phase.finish();
        std::hint::black_box(allocation);
    }

    /// Exact serial ACKs prevent nested or mismatched phase boundaries.
    #[cfg(target_os = "macos")]
    #[test]
    fn memory_channel_enforces_identity_and_order() {
        use std::io::{BufRead, BufReader, Write};
        let (client, mut server) = std::os::unix::net::UnixStream::pair().unwrap();
        let responder = std::thread::spawn(move || {
            let mut reader = BufReader::new(server.try_clone().unwrap());
            for action in ["idle", "begin", "end"] {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let event: serde_json::Value = serde_json::from_str(&line).unwrap();
                assert_eq!(event["event"], action);
                assert_eq!(event["id"], 0);
                let ack = serde_json::json!({"ok":true,"id":0,"event":action});
                writeln!(server, "{ack}").unwrap();
            }
        });
        let mut channel = MemoryChannel {
            stream: client,
            next_id: 0,
            active: None,
        };
        let id = channel.begin("warm_source_128_decode_128").unwrap();
        assert!(channel.begin("nested").is_err());
        assert!(channel.finish(id + 1).is_err());
        channel.finish(id).unwrap();
        assert_eq!(channel.next_id, 1);
        assert!(channel.active.is_none());
        responder.join().unwrap();
    }

    /// Stale sockets, malformed ACKs, EOF and timeouts fail closed without hanging.
    #[cfg(target_os = "macos")]
    #[test]
    fn memory_channel_failure_boundaries_are_bounded() {
        use std::io::{Read, Write};
        let directory = tempfile::tempdir().unwrap();
        assert!(MemoryChannel::connect(&directory.path().join("stale.sock")).is_err());
        let stale = directory.path().join("stale.sock");
        drop(std::os::unix::net::UnixListener::bind(&stale).unwrap());
        assert!(MemoryChannel::connect(&stale).is_err());
        for reply in [b"{}\n".to_vec(), vec![b'x'; 4097]] {
            let (client, mut server) = std::os::unix::net::UnixStream::pair().unwrap();
            let responder = std::thread::spawn(move || {
                let mut byte = [0_u8; 1];
                loop {
                    server.read_exact(&mut byte).unwrap();
                    if byte[0] == b'\n' {
                        break;
                    }
                }
                server.write_all(&reply).unwrap();
            });
            let mut channel = MemoryChannel {
                stream: client,
                next_id: 0,
                active: None,
            };
            assert!(channel.begin("strict_load").is_err());
            responder.join().unwrap();
        }
        let (client, _silent_peer) = std::os::unix::net::UnixStream::pair().unwrap();
        client
            .set_read_timeout(Some(std::time::Duration::from_millis(10)))
            .unwrap();
        let mut channel = MemoryChannel {
            stream: client,
            next_id: 0,
            active: None,
        };
        assert!(channel.begin("timeout").is_err());
        for invalid in ["", "contains space", &"x".repeat(129)] {
            assert!(channel.begin(invalid).is_err());
        }
        channel.next_id = u64::MAX;
        assert!(channel.begin("overflow").is_err());
        let (client, peer) = std::os::unix::net::UnixStream::pair().unwrap();
        drop(peer);
        let mut channel = MemoryChannel {
            stream: client,
            next_id: 0,
            active: None,
        };
        assert!(channel.begin("closed").is_err());
    }

    /// Median interpolation and nearest-rank p95 remain fixed for both odd and even counts.
    #[test]
    fn quantiles_preserve_raw_pairing_and_valid_work() {
        let samples = [3.0, 1.0, 4.0, 2.0, 5.0, 8.0, 7.0, 9.0, 6.0, 10.0];
        let result = summarize(&samples, Some(128), 10).unwrap();
        assert_eq!(result.samples_seconds, samples);
        assert_eq!(result.median_seconds, 5.5);
        assert_eq!(result.p95_seconds, 10.0);
        assert_eq!(result.minimum_seconds, 1.0);
        assert_eq!(result.maximum_seconds, 10.0);
        assert_eq!(result.tokens_per_second.unwrap()[1], 128.0);
        assert_eq!(
            summarize(&[2.0, 1.0, 3.0], None, 1).unwrap().median_seconds,
            2.0
        );
        assert_eq!(
            summarize(&(1..=20).map(f64::from).collect::<Vec<_>>(), None, 1)
                .unwrap()
                .p95_seconds,
            19.0
        );
    }

    /// Missing, non-finite and zero-duration samples cannot become attractive throughput values.
    #[test]
    fn invalid_metric_inputs_fail_closed() {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(summarize(&[value], None, 1).is_err());
        }
        assert!(summarize(&[], None, 1).is_err());
        assert!(summarize(&[1.0], None, 2).is_err());
        assert!(summarize(&[1.0], None, 0).is_err());
        assert!(summarize(&[1.0], Some(0), 1).is_err());
        assert!(summarize(&[f64::MIN_POSITIVE / 2.0], Some(usize::MAX), 1).is_err());
    }

    /// Fixed protocol counts and required environment facts cannot silently disappear.
    #[test]
    fn protocol_and_environment_require_complete_evidence() {
        assert_eq!(WARMUP_REPETITIONS, 3);
        assert_eq!(MEASURED_REPETITIONS, 10);
        assert_eq!(DECODE_STEPS, 128);
        let mut environment: RunEnvironment = serde_json::from_value(serde_json::json!({
            "machine_model":"test host", "cpu":"test CPU", "accelerator":"none",
            "physical_memory_bytes":1024, "operating_system":"test OS", "driver_runtime":"none",
            "power_mode":"test", "accelerator_exclusive":false,"other_workloads":"none","command":"test"
        })).unwrap();
        environment.validate().unwrap();
        environment.power_mode.clear();
        assert!(environment.validate().is_err());
    }

    /// Payload accounting includes preallocated capacity and a separate source-dependent bank.
    #[test]
    fn flan_cache_arithmetic_and_output_hashes_are_canonical() {
        let result = t5_cache_accounting(8, 6, 64, 512, 4, 1, 129, 512).unwrap();
        assert_eq!(result.self_kv_bytes_per_token_per_sequence, 24_576);
        assert_eq!(result.self_kv_allocated_bytes, 3_170_304);
        assert_eq!(result.cross_kv_bytes, 12_582_912);
        assert_eq!(result.retained_encoder_hidden_bytes, 1_048_576);
        assert_eq!(result.total_tensor_payload_bytes, 16_801_792);
        assert!(t5_cache_accounting(u64::MAX, 6, 64, 512, 4, 1, 129, 512).is_err());
        assert!(t5_cache_accounting(0, 6, 64, 512, 4, 1, 129, 512).is_err());
        assert_eq!(
            token_hash(&[1, 2]),
            "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f"
        );
        assert_ne!(token_hash(&[1, 2]), token_hash(&[2, 1]));
    }
}
