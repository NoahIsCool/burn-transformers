//! Muse forward, numerical-oracle, autodiff, and opt-in Metal attention tests.

#[cfg(all(feature = "metal", target_os = "macos"))]
use std::sync::Mutex;
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[cfg(all(feature = "metal", target_os = "macos"))]
use burn::backend::Metal;
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
use burn::tensor::module::attention;
use burn::{
    backend::Flex,
    module::Param,
    tensor::{Int, Tensor, TensorData, Tolerance, backend::BackendTypes, f16},
};
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
use tracing::{
    Event, Metadata, Subscriber,
    field::{Field, Visit},
    span::{Attributes, Id, Record},
};

use super::*;
#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::cache::{CacheConfig, KvCacheStorage, NativeKvCache, NativeKvCacheSpec};
use crate::{
    CausalLmInput, ForwardOptions, LogitsSelection, checkpoint::TensorSink,
    models::muse_glimmer::MuseGlimmerCheckpointOptions,
};
use burn::tensor::ElementConversion;

/// Deterministic CPU backend using F32 values and I32 token indices.
type TestBackend = Flex<f32, i32>;

/// Serializes the allocation-heavy 32K Metal envelope with other unit gates.
#[cfg(all(feature = "metal", target_os = "macos"))]
static MUSE_32K_METAL_STRESS_LOCK: Mutex<()> = Mutex::new(());

/// Supported synthetic stress length; this is not the checkpoint's 131K claim.
#[cfg(all(feature = "metal", target_os = "macos"))]
const MUSE_32K_STRESS_SEQUENCE: usize = 32_768;

/// Released Muse query-head count used by the synthetic Metal stress gate.
#[cfg(all(feature = "metal", target_os = "macos"))]
const MUSE_32K_STRESS_HEADS: usize = 32;

/// Released Muse native key/value-head count used by the stress gate.
#[cfg(all(feature = "metal", target_os = "macos"))]
const MUSE_32K_STRESS_KV_HEADS: usize = 2;

/// Reduced synthetic head width that retains QK, RoPE, GQA, and F16 semantics.
#[cfg(all(feature = "metal", target_os = "macos"))]
const MUSE_32K_STRESS_HEAD_DIM: usize = 4;

/// Released local-attention window used by Muse-Glimmer-30B.
#[cfg(all(feature = "metal", target_os = "macos"))]
const MUSE_32K_STRESS_WINDOW: usize = 2048;

/// Production ceiling shared by score, mask, and probability planes.
#[cfg(all(feature = "metal", target_os = "macos"))]
const MUSE_32K_STRESS_PLANE_LIMIT: usize = 1 << 26;

/// Serializes opt-in GPU work and process-local attention autotuning.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
static METAL_ATTENTION_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Sequence length that makes Burn 0.21 prioritize flash candidates over fallback.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
const METAL_ATTENTION_SEQUENCE: usize = 4097;

/// Pinned Muse-Glimmer attention head width.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
const METAL_ATTENTION_HEAD_DIM: usize = 128;

/// One CubeCL kernel submitted by the captured Metal attention call.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
#[derive(Clone, Debug, Eq, PartialEq)]
struct MetalKernelLaunch {
    /// Generated kernel name.
    name: String,
    /// Configuration-bearing kernel identifier.
    id: String,
}

/// Extracts the two display fields attached to CubeCL's launch span.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
#[derive(Default)]
struct MetalKernelFieldVisitor {
    /// Generated kernel name when present.
    name: Option<String>,
    /// Configuration-bearing kernel identifier when present.
    id: Option<String>,
}

#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
impl MetalKernelFieldVisitor {
    /// Stores one tracing field after removing debug-string quotes.
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

#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
impl Visit for MetalKernelFieldVisitor {
    /// Records a debug-formatted kernel field through the shared string decoder.
    fn record_debug(&mut self, field: &Field, value: &dyn core::fmt::Debug) {
        self.record_text(field, format!("{value:?}"));
    }

    /// Records a string-valued kernel field without debug formatting.
    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_text(field, value.to_owned());
    }
}

/// Thread-local tracing subscriber retaining launch names and identifiers.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
#[derive(Clone, Default)]
struct MetalKernelSubscriber {
    /// Launches shared with the capture caller.
    launches: Arc<Mutex<Vec<MetalKernelLaunch>>>,
    /// Non-zero span identifier source.
    next_span: Arc<AtomicU64>,
}

#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
impl Subscriber for MetalKernelSubscriber {
    /// Accepts only CubeCL launch spans for kernel-dispatch evidence.
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.name() == "launch_inner"
    }

    /// Stores complete kernel name/identifier pairs and assigns a nonzero span ID.
    fn new_span(&self, attributes: &Attributes<'_>) -> Id {
        let mut visitor = MetalKernelFieldVisitor::default();
        attributes.record(&mut visitor);
        if let (Some(name), Some(id)) = (visitor.name, visitor.id) {
            self.launches
                .lock()
                .expect("Metal kernel trace mutex must not be poisoned")
                .push(MetalKernelLaunch { name, id });
        }
        Id::from_u64(self.next_span.fetch_add(1, Ordering::Relaxed) + 1)
    }

    /// Ignores later field updates because launch metadata is captured when the span opens.
    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    /// Ignores span causality because launch identity is independent of relationships.
    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    /// Ignores standalone events; kernel evidence comes from launch spans.
    fn event(&self, _event: &Event<'_>) {}

    /// Ignores span entry because launch metadata was already recorded.
    fn enter(&self, _span: &Id) {}

    /// Ignores span exit because no timing or nesting information is retained.
    fn exit(&self, _span: &Id) {}
}

/// Captures every CubeCL launch submitted while one closure runs.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
fn capture_metal_kernel_launches<T>(operation: impl FnOnce() -> T) -> (T, Vec<MetalKernelLaunch>) {
    let subscriber = MetalKernelSubscriber::default();
    let launches = subscriber.launches.clone();
    let output = tracing::subscriber::with_default(subscriber, operation);
    let launches = Arc::try_unwrap(launches)
        .expect("capture must release its only launch-state owner")
        .into_inner()
        .expect("Metal kernel trace mutex must not be poisoned");
    (output, launches)
}

/// Identifies Burn 0.21's CubeCL flash-attention kernel by launch identity.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
fn is_flash_attention_launch(launch: &MetalKernelLaunch) -> bool {
    [&launch.name, &launch.id].into_iter().any(|identity| {
        let identity = identity.to_ascii_lowercase();
        identity.contains("cubek_attention") || identity.contains("flash_attention")
    })
}

/// Builds a four-layer miniature with the released local/local/local/global schedule.
fn tiny_config() -> MuseGlimmerTextConfig {
    MuseGlimmerTextConfig {
        vocab_size: 32,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 32,
        sliding_window: Some(3),
        bos_token_id: Some(crate::models::config::TokenIds::One(1)),
        eos_token_id: Some(crate::models::config::TokenIds::One(2)),
        ..Default::default()
    }
}

/// Installs deterministic small parameters without relying on process-global RNG state.
fn initialized_model<B: Backend>(
    config: MuseGlimmerTextConfig,
    device: &B::Device,
) -> MuseGlimmerForCausalLm<B> {
    let contract = config
        .checkpoint_contract(MuseGlimmerCheckpointOptions::default())
        .expect("tiny contract");
    let mut model = MuseGlimmerForCausalLm::init(config, device).expect("tiny model");
    for (tensor_index, expected) in contract.expected_tensors().iter().enumerate() {
        let shape = expected
            .shape
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        let elements = shape.iter().product();
        let values = (0..elements)
            .map(|element| {
                let code = (element + tensor_index * 5) % 29;
                (code as f32 - 14.0) * 0.003
            })
            .collect::<Vec<_>>();
        TensorSink::write_tensor(&mut model, &expected.name, TensorData::new(values, shape))
            .unwrap_or_else(|error| panic!("{}: {error}", expected.name));
    }
    model
}

/// Returns an initialized miniature and its device.
fn tiny_model() -> (
    MuseGlimmerForCausalLm<TestBackend>,
    <TestBackend as BackendTypes>::Device,
) {
    let device = Default::default();
    (initialized_model(tiny_config(), &device), device)
}

/// Builds an isolated attention fixture whose values expose its softmax weights.
///
/// The value rows are basis vectors and the output projection is the identity,
/// so the first three result channels equal the three attention probabilities.
/// A saturated gate retains them unchanged. This makes the fixture sensitive
/// to a one-ULP half-softmax regression without involving checkpoint loading.
fn softmax_oracle_attention(
    dtype: DType,
) -> (
    MuseGlimmerAttention<TestBackend>,
    MuseAttentionProjection<TestBackend>,
) {
    let device = Default::default();
    let config = MuseGlimmerTextConfig {
        hidden_size: 4,
        intermediate_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 4,
        max_position_embeddings: 8,
        sliding_window: Some(2),
        layer_types: vec![MuseGlimmerLayerType::FullAttention],
        layer_rope_theta: vec![0.0],
        vocab_size: 8,
        bos_token_id: None,
        eos_token_id: None,
        ..Default::default()
    };
    let mut attention = MuseGlimmerAttention::<TestBackend>::init(&config, &device).unwrap();
    attention.o_proj.weight = Param::from_tensor(
        Tensor::from_floats(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            &device,
        )
        .cast(dtype),
    );
    let projection = MuseAttentionProjection {
        // With head_dim=4 the implicit scale is 1/2, so these dot
        // products produce exact scores [-0.125, 5.875, 0.4375].
        query: Tensor::from_floats([[[[2.0, 0.0, 0.0, 0.0]]]], &device).cast(dtype),
        key: Tensor::<TestBackend, 3>::from_floats(
            [[
                [-0.125, 0.0, 0.0, 0.0],
                [5.875, 0.0, 0.0, 0.0],
                [0.4375, 0.0, 0.0, 0.0],
            ]],
            &device,
        )
        .unsqueeze_dim::<4>(0)
        .cast(dtype),
        value: Tensor::<TestBackend, 3>::from_floats(
            [[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]],
            &device,
        )
        .unsqueeze_dim::<4>(0)
        .cast(dtype),
        gate: Tensor::<TestBackend, 3>::full([1, 1, 4], 80.0, &device).cast(dtype),
    };
    (attention, projection)
}

/// F32 attention matches the literal upstream eager-softmax oracle.
///
/// The expected probabilities were generated from
/// `eager_attention_forward` in Hugging Face Transformers revision
/// `cfa309cfa34b5497a32fd50af49aa97aac94c054`. Keeping literals here makes
/// the test independent of Burn's own softmax implementation.
#[test]
fn attention_matches_pinned_fp32_softmax_oracle() {
    let (attention, projection) = softmax_oracle_attention(DType::F32);
    let actual = attention
        .attend(projection, None, false)
        .expect("valid isolated attention");
    let expected = TensorData::new(
        vec![0.002_461_939_6_f32, 0.993_217_3, 0.004_320_838_5, 0.0],
        [1, 1, 4],
    );
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected, Tolerance::absolute(1e-7));
}

/// F16 attention rounds the pinned FP32-softmax result, not a half-softmax.
///
/// The third expected half has bits `0x1c6d`; evaluating exponentiation and
/// normalization in F16 instead produces adjacent value `0x1c6c`. This
/// one-ULP discriminator directly covers Muse's explicit upstream
/// `softmax(..., dtype=torch.float32).to(query.dtype)` requirement at
/// Transformers revision `cfa309cfa34b5497a32fd50af49aa97aac94c054`.
#[test]
fn f16_attention_uses_fp32_softmax_before_rounding() {
    let (attention, projection) = softmax_oracle_attention(DType::F16);
    let actual = attention
        .attend(projection, None, false)
        .expect("valid isolated attention");
    assert_eq!(actual.dtype(), DType::F16);
    let bits = actual
        .into_data()
        .to_vec::<f16>()
        .expect("F16 result")
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(bits, [0x190b, 0x3bf2, 0x1c6d, 0x0000]);
    assert_ne!(bits[2], 0x1c6c, "half-precision softmax sentinel");
}

/// Builds the reduced-width configuration used only by the 32K Metal gate.
///
/// Query and KV head counts, context, window, GQA ratio, dtype boundary,
/// masks, and cache positions remain checkpoint-exact. Reducing the head
/// width from 128 to four removes irrelevant QK multiply work while leaving
/// the `[batch, head, query, key]` workspace dimensions unchanged.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_32k_stress_config() -> MuseGlimmerTextConfig {
    let hidden_size = MUSE_32K_STRESS_HEADS * MUSE_32K_STRESS_HEAD_DIM;
    MuseGlimmerTextConfig {
        vocab_size: 8,
        hidden_size,
        intermediate_size: hidden_size * 2,
        num_hidden_layers: 1,
        num_attention_heads: MUSE_32K_STRESS_HEADS,
        num_key_value_heads: MUSE_32K_STRESS_KV_HEADS,
        head_dim: MUSE_32K_STRESS_HEAD_DIM,
        max_position_embeddings: MUSE_32K_STRESS_SEQUENCE,
        sliding_window: Some(MUSE_32K_STRESS_WINDOW),
        layer_types: vec![MuseGlimmerLayerType::SlidingAttention],
        layer_rope_theta: vec![500_000.0],
        bos_token_id: None,
        eos_token_id: None,
        ..Default::default()
    }
}

/// Initializes synthetic official-head-count attention with an F16 identity output.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_32k_stress_attention(
    config: &MuseGlimmerTextConfig,
    device: &<Metal as BackendTypes>::Device,
) -> MuseGlimmerAttention<Metal> {
    let mut attention = MuseGlimmerAttention::<Metal>::init(config, device)
        .expect("valid synthetic Muse attention");
    let mut identity = vec![0.0_f32; config.hidden_size * config.hidden_size];
    for diagonal in 0..config.hidden_size {
        identity[diagonal * config.hidden_size + diagonal] = 1.0;
    }
    attention.o_proj.weight = Param::from_tensor(
        Tensor::<Metal, 2>::from_data(
            TensorData::new(identity, [config.hidden_size, config.hidden_size]),
            device,
        )
        .cast(DType::F16),
    );
    attention
}

/// Creates a 32K-key global suffix whose last row is an eager-F16 sentinel.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_32k_global_projection(
    query_sequence: usize,
    device: &<Metal as BackendTypes>::Device,
) -> MuseAttentionProjection<Metal> {
    let hidden_size = MUSE_32K_STRESS_HEADS * MUSE_32K_STRESS_HEAD_DIM;
    let mut query =
        vec![0.0_f32; MUSE_32K_STRESS_HEADS * query_sequence * MUSE_32K_STRESS_HEAD_DIM];
    for head in 0..MUSE_32K_STRESS_HEADS {
        for row in 0..query_sequence {
            query[(head * query_sequence + row) * MUSE_32K_STRESS_HEAD_DIM] = 2.0;
        }
    }
    let mut key =
        vec![
            0.0_f32;
            MUSE_32K_STRESS_KV_HEADS * MUSE_32K_STRESS_SEQUENCE * MUSE_32K_STRESS_HEAD_DIM
        ];
    let mut value = key.clone();
    for head in 0..MUSE_32K_STRESS_KV_HEADS {
        for row in 0..MUSE_32K_STRESS_SEQUENCE {
            key[(head * MUSE_32K_STRESS_SEQUENCE + row) * MUSE_32K_STRESS_HEAD_DIM] = -100.0;
        }
        for (offset, score) in [-0.125_f32, 5.875, 0.4375].into_iter().enumerate() {
            let row = MUSE_32K_STRESS_SEQUENCE - 3 + offset;
            key[(head * MUSE_32K_STRESS_SEQUENCE + row) * MUSE_32K_STRESS_HEAD_DIM] = score;
            value[(head * MUSE_32K_STRESS_SEQUENCE + row) * MUSE_32K_STRESS_HEAD_DIM + offset] =
                1.0;
        }
    }
    MuseAttentionProjection {
        query: Tensor::<Metal, 4>::from_data(
            TensorData::new(
                query,
                [
                    1,
                    MUSE_32K_STRESS_HEADS,
                    query_sequence,
                    MUSE_32K_STRESS_HEAD_DIM,
                ],
            ),
            device,
        )
        .cast(DType::F16),
        key: Tensor::<Metal, 4>::from_data(
            TensorData::new(
                key,
                [
                    1,
                    MUSE_32K_STRESS_KV_HEADS,
                    MUSE_32K_STRESS_SEQUENCE,
                    MUSE_32K_STRESS_HEAD_DIM,
                ],
            ),
            device,
        )
        .cast(DType::F16),
        value: Tensor::<Metal, 4>::from_data(
            TensorData::new(
                value,
                [
                    1,
                    MUSE_32K_STRESS_KV_HEADS,
                    MUSE_32K_STRESS_SEQUENCE,
                    MUSE_32K_STRESS_HEAD_DIM,
                ],
            ),
            device,
        )
        .cast(DType::F16),
        gate: Tensor::<Metal, 3>::full([1, query_sequence, hidden_size], 80.0, device)
            .cast(DType::F16),
    }
}

/// Creates terminal 32K-key local attention with three window-boundary probes.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_32k_sliding_projection(
    query_sequence: usize,
    device: &<Metal as BackendTypes>::Device,
) -> MuseAttentionProjection<Metal> {
    let hidden_size = MUSE_32K_STRESS_HEADS * MUSE_32K_STRESS_HEAD_DIM;
    let query = vec![0.0_f32; MUSE_32K_STRESS_HEADS * query_sequence * MUSE_32K_STRESS_HEAD_DIM];
    let key = vec![
        0.0_f32;
        MUSE_32K_STRESS_KV_HEADS * MUSE_32K_STRESS_SEQUENCE * MUSE_32K_STRESS_HEAD_DIM
    ];
    let mut value = key.clone();
    let query_offset = MUSE_32K_STRESS_SEQUENCE - query_sequence;
    let first_visible = query_offset - (MUSE_32K_STRESS_WINDOW - 1);
    let last_visible_start = MUSE_32K_STRESS_SEQUENCE - MUSE_32K_STRESS_WINDOW;
    for head in 0..MUSE_32K_STRESS_KV_HEADS {
        for (row, channel) in [
            (first_visible, 0usize),
            (last_visible_start, 1usize),
            (MUSE_32K_STRESS_SEQUENCE - 1, 2usize),
        ] {
            value[(head * MUSE_32K_STRESS_SEQUENCE + row) * MUSE_32K_STRESS_HEAD_DIM + channel] =
                1.0;
        }
    }
    MuseAttentionProjection {
        query: Tensor::<Metal, 4>::from_data(
            TensorData::new(
                query,
                [
                    1,
                    MUSE_32K_STRESS_HEADS,
                    query_sequence,
                    MUSE_32K_STRESS_HEAD_DIM,
                ],
            ),
            device,
        )
        .cast(DType::F16),
        key: Tensor::<Metal, 4>::from_data(
            TensorData::new(
                key,
                [
                    1,
                    MUSE_32K_STRESS_KV_HEADS,
                    MUSE_32K_STRESS_SEQUENCE,
                    MUSE_32K_STRESS_HEAD_DIM,
                ],
            ),
            device,
        )
        .cast(DType::F16),
        value: Tensor::<Metal, 4>::from_data(
            TensorData::new(
                value,
                [
                    1,
                    MUSE_32K_STRESS_KV_HEADS,
                    MUSE_32K_STRESS_SEQUENCE,
                    MUSE_32K_STRESS_HEAD_DIM,
                ],
            ),
            device,
        )
        .cast(DType::F16),
        gate: Tensor::<Metal, 3>::full([1, query_sequence, hidden_size], 80.0, device)
            .cast(DType::F16),
    }
}

/// Returns four F16 output channels for one query head and row.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_32k_head_bits(values: &[f16], row: usize, head: usize) -> [u16; 4] {
    let hidden_size = MUSE_32K_STRESS_HEADS * MUSE_32K_STRESS_HEAD_DIM;
    let start = row * hidden_size + head * MUSE_32K_STRESS_HEAD_DIM;
    std::array::from_fn(|channel| values[start + channel].to_bits())
}

/// Executes the bounded 32K Muse allocation envelope on Apple Metal.
///
/// This intentionally ignored hardware gate is synthetic, not 30B
/// checkpoint acceptance. It keeps the released 32/2 GQA topology, 2048
/// window, 32K absolute indices, F16 score/F32-softmax boundary, and exact
/// score-plane dimensions, while reducing head width to four and omitting
/// the dense decoder/FFN stack. It enumerates every full-context and local
/// production chunk, then executes worst-case global and local suffixes.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires a Metal adapter and roughly one GiB of transient workspace"]
fn stress_muse_32k_attention_envelope_on_metal() {
    let _guard = MUSE_32K_METAL_STRESS_LOCK
        .lock()
        .expect("32K Metal stress mutex must not be poisoned");
    let device = Default::default();
    let probe = Tensor::<Metal, 1>::zeros([1], &device)
        .into_data()
        .to_vec::<f32>()
        .expect("Metal adapter must expose F32 storage");
    assert_eq!(probe, [0.0]);

    let policy = AttentionExecutionPolicy::default();
    let global_plan = policy
        .plan(
            1,
            MUSE_32K_STRESS_HEADS,
            MUSE_32K_STRESS_SEQUENCE,
            MUSE_32K_STRESS_SEQUENCE,
        )
        .expect("32K global plan");
    let global_chunks = plan_muse_attention_chunks(
        policy,
        global_plan,
        1,
        MUSE_32K_STRESS_HEADS,
        MUSE_32K_STRESS_SEQUENCE,
        MUSE_32K_STRESS_SEQUENCE,
        None,
    )
    .expect("audit all global chunks");
    assert_eq!(global_chunks.len(), 512);
    assert!(
        global_chunks
            .iter()
            .all(|chunk| chunk.plane_elements == MUSE_32K_STRESS_PLANE_LIMIT)
    );
    assert_contiguous_query_coverage(&global_chunks, MUSE_32K_STRESS_SEQUENCE);

    let local_key_bound =
        MUSE_32K_STRESS_WINDOW + policy.maximum_query_tokens(MUSE_32K_STRESS_SEQUENCE) - 1;
    let local_plan = policy
        .plan(
            1,
            MUSE_32K_STRESS_HEADS,
            MUSE_32K_STRESS_SEQUENCE,
            local_key_bound,
        )
        .expect("32K sliding plan");
    let local_chunks = plan_muse_attention_chunks(
        policy,
        local_plan,
        1,
        MUSE_32K_STRESS_HEADS,
        MUSE_32K_STRESS_SEQUENCE,
        MUSE_32K_STRESS_SEQUENCE,
        Some(MUSE_32K_STRESS_WINDOW),
    )
    .expect("audit all sliding chunks");
    assert_eq!(local_chunks.len(), 64);
    assert_eq!(
        local_chunks.iter().map(|chunk| chunk.plane_elements).max(),
        Some(41_926_656)
    );
    assert!(
        local_chunks
            .iter()
            .all(|chunk| chunk.plane_elements <= MUSE_32K_STRESS_PLANE_LIMIT)
    );
    assert_contiguous_query_coverage(&local_chunks, MUSE_32K_STRESS_SEQUENCE);

    let config = metal_32k_stress_config();
    let token_mask = TokenMask::new(Tensor::<Metal, 2, Bool>::from_data(
        TensorData::new(
            vec![true; MUSE_32K_STRESS_SEQUENCE],
            [1, MUSE_32K_STRESS_SEQUENCE],
        ),
        &device,
    ));
    let position_input = CausalLmInput {
        token_ids: Tensor::<Metal, 2, Int>::zeros([1, MUSE_32K_STRESS_SEQUENCE], &device),
        token_mask: Some(token_mask),
        position_ids: None,
    };
    let validated = validate_input(
        &position_input,
        &device,
        &InputContract::new(INPUT_NAMES, PARAMETERS, config.vocab_size)
            .with_position_limit(MUSE_32K_STRESS_SEQUENCE),
    )
    .expect("position 32767 is inside the 32K capacity");
    assert_eq!(validated.evaluated_context, MUSE_32K_STRESS_SEQUENCE);
    let positions = validated
        .require_positions(INPUT_NAMES.position_ids)
        .expect("derive all 32K valid-token positions");
    let position_edges = Tensor::cat(
        vec![
            positions.clone().slice([0..1, 0..1]),
            positions
                .clone()
                .slice([0..1, MUSE_32K_STRESS_SEQUENCE - 1..MUSE_32K_STRESS_SEQUENCE]),
        ],
        1,
    )
    .into_data()
    .to_vec::<i32>()
    .expect("read 32K position edges");
    assert_eq!(position_edges, [0, 32_767]);
    let angles = rotary_angles(
        muse_rope_config(&config, 500_000.0).expect("synthetic Muse RoPE"),
        positions,
        MUSE_32K_STRESS_SEQUENCE,
    )
    .expect("build 32K Metal RoPE tables");
    assert_eq!(
        angles.cosine.dims(),
        [1, 1, MUSE_32K_STRESS_SEQUENCE, MUSE_32K_STRESS_HEAD_DIM]
    );
    let terminal_angles = Tensor::cat(
        vec![
            angles.cosine.slice([
                0..1,
                0..1,
                MUSE_32K_STRESS_SEQUENCE - 1..MUSE_32K_STRESS_SEQUENCE,
                0..MUSE_32K_STRESS_HEAD_DIM,
            ]),
            angles.sine.slice([
                0..1,
                0..1,
                MUSE_32K_STRESS_SEQUENCE - 1..MUSE_32K_STRESS_SEQUENCE,
                0..MUSE_32K_STRESS_HEAD_DIM,
            ]),
        ],
        3,
    )
    .into_data()
    .to_vec::<f32>()
    .expect("read terminal RoPE angles");
    assert!(terminal_angles.iter().all(|value| value.is_finite()));
    drop(position_input);
    <Metal as Backend>::sync(&device).expect("synchronize position checks");
    <Metal as Backend>::memory_cleanup(&device);

    let full_cache_config =
        CacheConfig::new(1, MUSE_32K_STRESS_SEQUENCE).expect("32K full cache config");
    let cache_spec = NativeKvCacheSpec::new(
        1,
        MUSE_32K_STRESS_KV_HEADS,
        MUSE_32K_STRESS_HEAD_DIM,
        KvCacheStorage::Full,
        DType::F16,
    )
    .expect("full cache spec");
    let mut full_cache = NativeKvCache::<Metal>::new(full_cache_config, 1, cache_spec, &device)
        .expect("allocate 32K global KV cache");
    let full_tail = Tensor::<Metal, 4>::full(
        [1, MUSE_32K_STRESS_KV_HEADS, 1, MUSE_32K_STRESS_HEAD_DIM],
        7.0,
        &device,
    )
    .cast(DType::F16);
    full_cache
        .write_sequence(
            0,
            0,
            MUSE_32K_STRESS_SEQUENCE - 1,
            full_tail.clone(),
            full_tail,
        )
        .expect("write global cache position 32767");
    let full_read = full_cache
        .read_sequence(0, 0, MUSE_32K_STRESS_SEQUENCE)
        .expect("read complete 32K global cache");
    assert_eq!(full_read.retained_positions(), 0..MUSE_32K_STRESS_SEQUENCE);
    let (full_keys, _) = full_read.materialize().expect("nonempty full cache");
    assert_eq!(
        full_keys.dims(),
        [
            1,
            MUSE_32K_STRESS_KV_HEADS,
            MUSE_32K_STRESS_SEQUENCE,
            MUSE_32K_STRESS_HEAD_DIM,
        ]
    );
    let full_tail_value = full_keys
        .slice([
            0..1,
            0..1,
            MUSE_32K_STRESS_SEQUENCE - 1..MUSE_32K_STRESS_SEQUENCE,
            0..1,
        ])
        .into_data()
        .to_vec::<f16>()
        .expect("read global cache tail");
    assert_eq!(full_tail_value[0].to_f32(), 7.0);

    let ring_cache_config = CacheConfig::new(1, MUSE_32K_STRESS_SEQUENCE)
        .expect("32K ring cache config")
        .with_sliding_window(MUSE_32K_STRESS_WINDOW)
        .expect("released sliding window");
    let ring_spec = NativeKvCacheSpec::new(
        1,
        MUSE_32K_STRESS_KV_HEADS,
        MUSE_32K_STRESS_HEAD_DIM,
        KvCacheStorage::MirroredSlidingRing,
        DType::F16,
    )
    .expect("mirrored ring cache spec");
    let mut ring_cache = NativeKvCache::<Metal>::new(ring_cache_config, 1, ring_spec, &device)
        .expect("allocate 2048-token mirrored ring");
    let mut ring_tail = vec![0.0_f32; MUSE_32K_STRESS_KV_HEADS * 2 * MUSE_32K_STRESS_HEAD_DIM];
    for head in 0..MUSE_32K_STRESS_KV_HEADS {
        for channel in 0..MUSE_32K_STRESS_HEAD_DIM {
            ring_tail[(head * 2) * MUSE_32K_STRESS_HEAD_DIM + channel] = 11.0;
            ring_tail[(head * 2 + 1) * MUSE_32K_STRESS_HEAD_DIM + channel] = 13.0;
        }
    }
    let ring_tail = Tensor::<Metal, 4>::from_data(
        TensorData::new(
            ring_tail,
            [1, MUSE_32K_STRESS_KV_HEADS, 2, MUSE_32K_STRESS_HEAD_DIM],
        ),
        &device,
    )
    .cast(DType::F16);
    ring_cache
        .write_sequence(
            0,
            0,
            MUSE_32K_STRESS_SEQUENCE - 2,
            ring_tail.clone(),
            ring_tail,
        )
        .expect("write mirrored positions 32766 and 32767");
    let ring_read = ring_cache
        .read_sequence(0, 0, MUSE_32K_STRESS_SEQUENCE)
        .expect("read terminal 2048-token window");
    assert_eq!(
        ring_read.retained_positions(),
        MUSE_32K_STRESS_SEQUENCE - MUSE_32K_STRESS_WINDOW..MUSE_32K_STRESS_SEQUENCE
    );
    assert_eq!(ring_read.segments().len(), 1);
    assert_eq!(
        ring_read.segments()[0].absolute_positions(),
        MUSE_32K_STRESS_SEQUENCE - MUSE_32K_STRESS_WINDOW..MUSE_32K_STRESS_SEQUENCE
    );
    let (ring_keys, _) = ring_read.materialize().expect("nonempty ring cache");
    assert_eq!(
        ring_keys.dims(),
        [
            1,
            MUSE_32K_STRESS_KV_HEADS,
            MUSE_32K_STRESS_WINDOW,
            MUSE_32K_STRESS_HEAD_DIM,
        ]
    );
    let ring_tail_values = ring_keys
        .slice([
            0..1,
            0..1,
            MUSE_32K_STRESS_WINDOW - 2..MUSE_32K_STRESS_WINDOW,
            0..1,
        ])
        .into_data()
        .to_vec::<f16>()
        .expect("read wrapped ring tail")
        .into_iter()
        .map(f16::to_f32)
        .collect::<Vec<_>>();
    assert_eq!(ring_tail_values, [11.0, 13.0]);
    drop(full_read);
    drop(full_cache);
    drop(ring_read);
    drop(ring_cache);
    <Metal as Backend>::sync(&device).expect("synchronize cache checks");
    <Metal as Backend>::memory_cleanup(&device);

    let attention = metal_32k_stress_attention(&config, &device);
    /// Global suffix length crossing the policy's 64-row chunk boundary.
    const GLOBAL_QUERY_ROWS: usize = 65;
    let representative_global_plan = policy
        .plan(
            1,
            MUSE_32K_STRESS_HEADS,
            GLOBAL_QUERY_ROWS,
            MUSE_32K_STRESS_SEQUENCE,
        )
        .expect("representative global suffix plan");
    let representative_global_chunks = plan_muse_attention_chunks(
        policy,
        representative_global_plan,
        1,
        MUSE_32K_STRESS_HEADS,
        GLOBAL_QUERY_ROWS,
        MUSE_32K_STRESS_SEQUENCE,
        None,
    )
    .expect("audit representative global suffix");
    assert_eq!(
        representative_global_chunks
            .iter()
            .map(|chunk| chunk.plane_shape)
            .collect::<Vec<_>>(),
        [
            [1, MUSE_32K_STRESS_HEADS, 64, MUSE_32K_STRESS_SEQUENCE,],
            [1, MUSE_32K_STRESS_HEADS, 1, MUSE_32K_STRESS_SEQUENCE,],
        ]
    );
    let global_projection = metal_32k_global_projection(GLOBAL_QUERY_ROWS, &device);
    assert_eq!(
        muse_attention_precision(global_projection.query.dtype()),
        AttentionPrecisionMode::EagerFp32Softmax
    );
    let global_output = attention
        .attend_cached(global_projection, None)
        .expect("execute bounded 32K-key global attention on Metal");
    assert_eq!(global_output.dtype(), DType::F16);
    let global_values = global_output
        .into_data()
        .to_vec::<f16>()
        .expect("read global F16 output");
    assert_eq!(metal_32k_head_bits(&global_values, 0, 0), [0; 4]);
    for head in [0, 15, 16, 31] {
        let bits = metal_32k_head_bits(&global_values, GLOBAL_QUERY_ROWS - 1, head);
        assert_eq!(bits, [0x190b, 0x3bf2, 0x1c6d, 0x0000]);
        assert_ne!(bits[2], 0x1c6c, "must not run F16 softmax");
    }
    drop(global_values);
    <Metal as Backend>::sync(&device).expect("synchronize global stress attention");
    <Metal as Backend>::memory_cleanup(&device);

    /// Local suffix length spanning multiple bounded attention chunks.
    const LOCAL_QUERY_ROWS: usize = 513;
    let representative_local_bound =
        MUSE_32K_STRESS_WINDOW + policy.maximum_query_tokens(LOCAL_QUERY_ROWS) - 1;
    let representative_local_plan = policy
        .plan(
            1,
            MUSE_32K_STRESS_HEADS,
            LOCAL_QUERY_ROWS,
            representative_local_bound,
        )
        .expect("representative sliding suffix plan");
    let representative_local_chunks = plan_muse_attention_chunks(
        policy,
        representative_local_plan,
        1,
        MUSE_32K_STRESS_HEADS,
        LOCAL_QUERY_ROWS,
        MUSE_32K_STRESS_SEQUENCE,
        Some(MUSE_32K_STRESS_WINDOW),
    )
    .expect("audit representative sliding suffix");
    assert_eq!(representative_local_chunks.len(), 2);
    assert_eq!(representative_local_chunks[0].key_range, 30_208..32_767);
    assert_eq!(
        representative_local_chunks[0].plane_shape,
        [1, MUSE_32K_STRESS_HEADS, 512, 2559]
    );
    assert_eq!(representative_local_chunks[1].key_range, 30_720..32_768);
    assert_eq!(
        representative_local_chunks[1].plane_shape,
        [1, MUSE_32K_STRESS_HEADS, 1, 2048]
    );
    let local_output = attention
        .attend_cached(
            metal_32k_sliding_projection(LOCAL_QUERY_ROWS, &device),
            Some(MUSE_32K_STRESS_WINDOW),
        )
        .expect("execute bounded 32K-key sliding attention on Metal");
    assert_eq!(local_output.dtype(), DType::F16);
    let local_values = local_output
        .into_data()
        .to_vec::<f16>()
        .expect("read local F16 output");
    for head in [0, 15, 16, 31] {
        assert_eq!(
            metal_32k_head_bits(&local_values, 0, head),
            [0x1000, 0x1000, 0x0000, 0x0000]
        );
        assert_eq!(
            metal_32k_head_bits(&local_values, LOCAL_QUERY_ROWS - 1, head),
            [0x0000, 0x1000, 0x1000, 0x0000]
        );
    }
    <Metal as Backend>::sync(&device).expect("synchronize sliding stress attention");
    eprintln!(
        "synthetic Muse 32K Metal envelope passed: global={}x[1,32,64,32768], local={} chunks with peak {} elements; this is not full 30B 32K acceptance",
        global_chunks.len(),
        local_chunks.len(),
        local_chunks
            .iter()
            .map(|chunk| chunk.plane_elements)
            .max()
            .unwrap_or(0),
    );
}

/// Builds an official-width Metal fixture whose score rounding changes F16 softmax bits.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
fn metal_score_rounding_fixture(
    device: &<Metal as BackendTypes>::Device,
) -> (MuseGlimmerAttention<Metal>, MuseAttentionProjection<Metal>) {
    /// Single-head fixture isolating the eager score-rounding boundary.
    const HEADS: usize = 1;
    /// Residual width matching the identity attention projection.
    const HIDDEN_SIZE: usize = HEADS * METAL_ATTENTION_HEAD_DIM;
    /// Nonzero key rows controlling the last-query rounding oracle.
    const VISIBLE_KEYS: usize = 4;

    let config = MuseGlimmerTextConfig {
        hidden_size: HIDDEN_SIZE,
        intermediate_size: HIDDEN_SIZE * 2,
        num_hidden_layers: 1,
        num_attention_heads: HEADS,
        num_key_value_heads: HEADS,
        head_dim: METAL_ATTENTION_HEAD_DIM,
        max_position_embeddings: METAL_ATTENTION_SEQUENCE,
        sliding_window: None,
        layer_types: vec![MuseGlimmerLayerType::FullAttention],
        layer_rope_theta: vec![0.0],
        vocab_size: 8,
        bos_token_id: None,
        eos_token_id: None,
        ..Default::default()
    };
    let mut muse_attention = MuseGlimmerAttention::<Metal>::init(&config, device).unwrap();
    let mut identity = vec![0.0_f32; HIDDEN_SIZE * HIDDEN_SIZE];
    for diagonal in 0..HIDDEN_SIZE {
        identity[diagonal * HIDDEN_SIZE + diagonal] = 1.0;
    }
    muse_attention.o_proj.weight = Param::from_tensor(
        Tensor::<Metal, 2>::from_data(
            TensorData::new(identity, [HIDDEN_SIZE, HIDDEN_SIZE]),
            device,
        )
        .cast(DType::F16),
    );

    let mut query = vec![0.0_f32; HEADS * METAL_ATTENTION_SEQUENCE * METAL_ATTENTION_HEAD_DIM];
    let mut key = vec![0.0_f32; HEADS * METAL_ATTENTION_SEQUENCE * METAL_ATTENTION_HEAD_DIM];
    let mut value = vec![0.0_f32; HEADS * METAL_ATTENTION_SEQUENCE * METAL_ATTENTION_HEAD_DIM];
    for head in 0..HEADS {
        for row in 0..METAL_ATTENTION_SEQUENCE {
            query[(head * METAL_ATTENTION_SEQUENCE + row) * METAL_ATTENTION_HEAD_DIM] = 1.730_468_8;
            key[(head * METAL_ATTENTION_SEQUENCE + row) * METAL_ATTENTION_HEAD_DIM] = -100.0;
        }
        for (row, candidate) in [-19.593_75, 7.781_25, -24.937_5, 16.812_5]
            .into_iter()
            .enumerate()
        {
            key[(head * METAL_ATTENTION_SEQUENCE + row) * METAL_ATTENTION_HEAD_DIM] = candidate;
        }
        for row in 0..VISIBLE_KEYS {
            value[(head * METAL_ATTENTION_SEQUENCE + row) * METAL_ATTENTION_HEAD_DIM + row] = 1.0;
        }
    }

    let projection = MuseAttentionProjection {
        query: Tensor::<Metal, 4>::from_data(
            TensorData::new(
                query,
                [1, HEADS, METAL_ATTENTION_SEQUENCE, METAL_ATTENTION_HEAD_DIM],
            ),
            device,
        )
        .cast(DType::F16),
        key: Tensor::<Metal, 4>::from_data(
            TensorData::new(
                key,
                [1, HEADS, METAL_ATTENTION_SEQUENCE, METAL_ATTENTION_HEAD_DIM],
            ),
            device,
        )
        .cast(DType::F16),
        value: Tensor::<Metal, 4>::from_data(
            TensorData::new(
                value,
                [1, HEADS, METAL_ATTENTION_SEQUENCE, METAL_ATTENTION_HEAD_DIM],
            ),
            device,
        )
        .cast(DType::F16),
        gate: Tensor::<Metal, 3>::full([1, METAL_ATTENTION_SEQUENCE, HIDDEN_SIZE], 80.0, device)
            .cast(DType::F16),
    };
    (muse_attention, projection)
}

/// Evaluates the pinned eager score-rounding and FP32-softmax operation graph.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
fn pinned_eager_last_query(projection: &MuseAttentionProjection<Metal>) -> Tensor<Metal, 4> {
    let query = projection.query.clone().slice([
        0..1,
        0..1,
        METAL_ATTENTION_SEQUENCE - 1..METAL_ATTENTION_SEQUENCE,
        0..METAL_ATTENTION_HEAD_DIM,
    ]);
    let key = projection.key.clone().slice([
        0..1,
        0..1,
        0..METAL_ATTENTION_SEQUENCE,
        0..METAL_ATTENTION_HEAD_DIM,
    ]);
    let value = projection.value.clone().slice([
        0..1,
        0..1,
        0..METAL_ATTENTION_SEQUENCE,
        0..METAL_ATTENTION_HEAD_DIM,
    ]);
    let scores =
        query.matmul(key.swap_dims(2, 3)) * (METAL_ATTENTION_HEAD_DIM as f64).sqrt().recip();
    let probabilities =
        burn::tensor::activation::softmax(scores.cast(DType::F32), 3).cast(DType::F16);
    probabilities.matmul(value)
}

/// Evaluates flash attention's strict-F32 QK and softmax arithmetic for the sentinel.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
fn strict_f32_last_query(projection: &MuseAttentionProjection<Metal>) -> Tensor<Metal, 4> {
    let query = projection
        .query
        .clone()
        .slice([
            0..1,
            0..1,
            METAL_ATTENTION_SEQUENCE - 1..METAL_ATTENTION_SEQUENCE,
            0..METAL_ATTENTION_HEAD_DIM,
        ])
        .cast(DType::F32);
    let key = projection
        .key
        .clone()
        .slice([
            0..1,
            0..1,
            0..METAL_ATTENTION_SEQUENCE,
            0..METAL_ATTENTION_HEAD_DIM,
        ])
        .cast(DType::F32);
    let value = projection
        .value
        .clone()
        .slice([
            0..1,
            0..1,
            0..METAL_ATTENTION_SEQUENCE,
            0..METAL_ATTENTION_HEAD_DIM,
        ])
        .cast(DType::F32);
    let scores =
        query.matmul(key.swap_dims(2, 3)) * (METAL_ATTENTION_HEAD_DIM as f64).sqrt().recip();
    burn::tensor::activation::softmax(scores, 3)
        .matmul(value)
        .cast(DType::F16)
}

/// Reads the first four channels of one `[1, 1, 1, head_dim]` F16 result.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
fn first_four_f16_bits(tensor: Tensor<Metal, 4>) -> Vec<u16> {
    tensor
        .slice([0..1, 0..1, 0..1, 0..4])
        .into_data()
        .to_vec::<f16>()
        .expect("Metal result must retain F16 storage")
        .into_iter()
        .map(f16::to_bits)
        .collect()
}

/// Muse F16 bypasses flash when its strict-F32 score graph changes results.
///
/// Run this opt-in gate alone because it enables process-local autotuning:
/// `cargo test --manifest-path crates/transformers/Cargo.toml
/// --no-default-features --features std,flex,attention-kernel-tracing
/// --lib
/// models::muse_glimmer::model::tests::metal_f16_attention_matches_pinned_eager_score_rounding
/// -- --ignored --exact --nocapture --test-threads=1`.
#[cfg(all(feature = "attention-kernel-tracing", target_os = "macos"))]
#[test]
#[ignore = "requires a Metal-capable macOS host and serialized attention autotuning"]
fn metal_f16_attention_matches_pinned_eager_score_rounding() {
    let _serial = METAL_ATTENTION_TEST_LOCK
        .lock()
        .expect("Metal attention test lock must not be poisoned");
    let device = Default::default();
    let (muse_attention, projection) = metal_score_rounding_fixture(&device);

    let eager_bits = first_four_f16_bits(pinned_eager_last_query(&projection));
    let strict_bits = first_four_f16_bits(strict_f32_last_query(&projection));
    assert_ne!(
        eager_bits, strict_bits,
        "Metal fixture must distinguish the pinned F16 score boundary from strict F32"
    );

    let warmup = attention(
        projection.query.clone(),
        projection.key.clone(),
        projection.value.clone(),
        None,
        None,
        muse_attention_options(true),
    );
    <Metal<f32, i32> as Backend>::sync(&device).expect("synchronize attention autotuning");
    let _ = warmup.into_data();

    let (fused_bits, launches) = capture_metal_kernel_launches(|| {
        let output = attention(
            projection.query.clone(),
            projection.key.clone(),
            projection.value.clone(),
            None,
            None,
            muse_attention_options(true),
        );
        <Metal<f32, i32> as Backend>::sync(&device).expect("synchronize fused attention");
        first_four_f16_bits(output.slice([
            0..1,
            0..1,
            METAL_ATTENTION_SEQUENCE - 1..METAL_ATTENTION_SEQUENCE,
            0..METAL_ATTENTION_HEAD_DIM,
        ]))
    });
    let used_flash = launches.iter().any(is_flash_attention_launch);
    assert!(
        used_flash,
        "autotuned call must execute an attention kernel: {launches:#?}"
    );
    assert_eq!(
        fused_bits, strict_bits,
        "Burn 0.21 flash must exercise the strict-F32 discriminator"
    );
    assert_ne!(
        fused_bits, eager_bits,
        "fixture must expose the missing F16 score-rounding boundary"
    );

    let plan = muse_attention
        .execution_policy
        .plan(1, 1, METAL_ATTENTION_SEQUENCE, METAL_ATTENTION_SEQUENCE)
        .expect("default Muse policy must cover the Metal sentinel");
    assert_eq!(plan.query_chunk_size(), 512);
    assert_eq!(plan.chunk_count(), 9);
    let (muse_bits, muse_launches) = capture_metal_kernel_launches(|| {
        muse_attention
            .attend(projection, None, true)
            .expect("valid Muse Metal attention")
            .slice([
                0..1,
                METAL_ATTENTION_SEQUENCE - 1..METAL_ATTENTION_SEQUENCE,
                0..4,
            ])
            .into_data()
            .to_vec::<f16>()
            .expect("Muse output must retain F16 storage")
            .into_iter()
            .map(f16::to_bits)
            .collect::<Vec<_>>()
    });
    let muse_used_flash = muse_launches.iter().any(is_flash_attention_launch);
    assert!(
        !muse_used_flash,
        "the pinned eager Muse path must not silently dispatch flash: {muse_launches:#?}"
    );
    eprintln!(
        "Muse Metal score-rounding gate: eager={eager_bits:04x?}, strict={strict_bits:04x?}, fused={fused_bits:04x?}, used_flash={used_flash}, muse_used_flash={muse_used_flash}, muse_chunks={}x{}",
        plan.chunk_count(),
        plan.query_chunk_size(),
    );
    assert_eq!(muse_bits, eager_bits, "Muse must match pinned eager F16");
}

/// Non-F16 Muse options retain Burn's flash-attention eligibility contract.
///
/// Burn 0.21 CubeCL falls back whenever `scale.is_some()`, even when that
/// value redundantly equals `1/sqrt(head_dim)`. Its default Metal build also
/// chooses fallback because `autotune` is disabled; eligibility and runtime
/// selection are distinct. F16 Muse intentionally uses the explicit shared
/// eager mode before these options reach a backend kernel.
#[test]
fn attention_options_preserve_non_f16_flash_eligibility() {
    assert_eq!(
        muse_attention_options(true),
        AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: true,
        }
    );
    assert_eq!(muse_attention_options(false).scale, None);
}

/// Sliding query chunks retain precisely the union of row-local key ranges.
#[test]
fn aligned_local_key_ranges_respect_inclusive_window_boundary() {
    assert_eq!(
        aligned_key_range(32_768, 32_768, 0..512, 2048).unwrap(),
        0..512
    );
    assert_eq!(
        aligned_key_range(32_768, 32_768, 2048..2560, 2048).unwrap(),
        1..2560
    );
    assert_eq!(aligned_key_range(2, 5, 0..2, 3).unwrap(), 1..5);
    assert!(aligned_key_range(5, 4, 0..1, 3).is_err());
    assert!(aligned_key_range(5, 5, 0..1, 0).is_err());
}

/// The concrete 32K global and local schedules bound every backend plane.
///
/// This enumerates the exact schedule consumed by `attend_chunked`: no
/// device allocation is inferred from only the planner's peak. Scores,
/// masks, F32 softmax storage, and rounded probabilities all use each
/// recorded `[batch, head, query, key]` geometry.
#[test]
fn official_32k_chunk_schedules_bound_every_realized_plane() {
    /// Synthetic context length used to enumerate production chunk geometry.
    const SEQUENCE: usize = 32_768;
    /// Released query-head count retained by the chunk-geometry test.
    const HEADS: usize = 32;
    /// Released local-attention window measured in semantic tokens.
    const WINDOW: usize = 2048;
    /// Maximum permitted element count in each realized score-like plane.
    const LIMIT: usize = 1 << 26;

    let policy = AttentionExecutionPolicy::default();
    let global_plan = policy
        .plan(1, HEADS, SEQUENCE, SEQUENCE)
        .expect("official global plan");
    let global =
        plan_muse_attention_chunks(policy, global_plan, 1, HEADS, SEQUENCE, SEQUENCE, None)
            .expect("expand global chunks");
    assert_eq!(global.len(), 512);
    assert_eq!(global.first().unwrap().query_range, 0..64);
    assert_eq!(global.last().unwrap().query_range, 32_704..32_768);
    assert!(global.iter().all(|chunk| chunk.key_range == (0..SEQUENCE)));
    assert!(
        global
            .iter()
            .all(|chunk| chunk.plane_shape == [1, HEADS, 64, SEQUENCE])
    );
    assert!(global.iter().all(|chunk| chunk.plane_elements == LIMIT));
    assert_contiguous_query_coverage(&global, SEQUENCE);

    let local_key_bound = WINDOW + policy.maximum_query_tokens(SEQUENCE) - 1;
    let local_plan = policy
        .plan(1, HEADS, SEQUENCE, local_key_bound)
        .expect("official local plan");
    let local = plan_muse_attention_chunks(
        policy,
        local_plan,
        1,
        HEADS,
        SEQUENCE,
        SEQUENCE,
        Some(WINDOW),
    )
    .expect("expand local chunks");
    assert_eq!(local.len(), 64);
    assert_eq!(local[0].query_range, 0..512);
    assert_eq!(local[0].key_range, 0..512);
    assert_eq!(local[4].query_range, 2048..2560);
    assert_eq!(local[4].key_range, 1..2560);
    assert_eq!(local.last().unwrap().query_range, 32_256..32_768);
    assert_eq!(local.last().unwrap().key_range, 30_209..32_768);
    assert_eq!(
        local.iter().map(|chunk| chunk.plane_elements).max(),
        Some(41_926_656)
    );
    assert!(local.iter().all(|chunk| chunk.plane_elements <= LIMIT));
    assert!(
        local
            .iter()
            .all(|chunk| chunk.key_range.len() <= local_key_bound)
    );
    assert_contiguous_query_coverage(&local, SEQUENCE);
}

/// Concrete expansion rejects stale plans and underestimated local unions.
#[test]
fn chunk_schedule_fails_closed_on_plan_drift_and_key_underestimates() {
    let policy = AttentionExecutionPolicy::query_chunked(4, 1024);
    let plan = policy.plan(1, 2, 8, 4).expect("small plan");
    let query_drift = plan_muse_attention_chunks(policy, plan, 1, 2, 7, 8, None)
        .expect_err("query geometry must match the immutable plan");
    assert!(query_drift.to_string().contains("covers 8 query rows"));

    let underestimated = plan_muse_attention_chunks(policy, plan, 1, 2, 8, 8, Some(5))
        .expect_err("later local union is wider than the declared bound");
    assert!(
        underestimated
            .to_string()
            .contains("exceeds planned bound 4")
    );

    let plane_drift =
        plan_muse_attention_chunks(AttentionExecutionPolicy::Unbounded, plan, 1, 3, 8, 4, None)
            .expect_err("a looser runtime policy cannot widen an immutable plan");
    assert!(plane_drift.to_string().contains("immutable execution plan"));

    let empty_keys = plan_muse_attention_chunks(policy, plan, 1, 2, 8, 0, None)
        .expect_err("empty key storage must fail before slicing");
    assert!(
        empty_keys
            .to_string()
            .contains("key sequence must be non-zero")
    );
}

/// Verifies complete, gap-free query coverage for an audited chunk schedule.
fn assert_contiguous_query_coverage(chunks: &[MuseAttentionChunk], sequence: usize) {
    let mut expected_start = 0;
    for chunk in chunks {
        assert_eq!(chunk.query_range.start, expected_start);
        assert!(!chunk.query_range.is_empty());
        assert!(!chunk.key_range.is_empty());
        assert_eq!(
            chunk.plane_elements,
            chunk.plane_shape.into_iter().product::<usize>()
        );
        expected_start = chunk.query_range.end;
    }
    assert_eq!(expected_start, sequence);
}

/// Muse defaults to a bounded policy and rejects invalid runtime overrides.
#[test]
fn model_exposes_validated_attention_execution_policy() {
    let (model, _) = tiny_model();
    assert_eq!(
        model.attention_execution_policy(),
        AttentionExecutionPolicy::default()
    );
    let error = model
        .clone()
        .with_attention_execution_policy(AttentionExecutionPolicy::query_chunked(0, 1))
        .expect_err("zero query chunks are invalid");
    assert!(error.to_string().contains("max_query_tokens"));
    let unbounded = model
        .with_attention_execution_policy(AttentionExecutionPolicy::Unbounded)
        .expect("explicit unbounded policy");
    assert_eq!(
        unbounded.attention_execution_policy(),
        AttentionExecutionPolicy::Unbounded
    );
}

/// Query chunking retains full/sliding causal and padded-window semantics.
#[test]
fn bounded_stateless_attention_matches_unbounded_with_padding_holes() {
    let (model, device) = tiny_model();
    let unbounded = model
        .clone()
        .with_attention_execution_policy(AttentionExecutionPolicy::Unbounded)
        .expect("unbounded fixture");
    let bounded = model
        .with_attention_execution_policy(AttentionExecutionPolicy::query_chunked(2, 20))
        .expect("two-row chunks");
    let input = || CausalLmInput {
        token_ids: Tensor::from_ints([[1, 2, 3, 4, 5]], &device),
        token_mask: Some(TokenMask::new(Tensor::from_bool(
            TensorData::from([[true, false, true, true, true]]),
            &device,
        ))),
        position_ids: None,
    };
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let expected = unbounded
        .forward(input(), &options)
        .expect("unbounded forward")
        .logits;
    let actual = bounded
        .forward(input(), &options)
        .expect("bounded forward")
        .logits;
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(2e-6));
}

/// A redundant all-valid mask takes the optimized unpadded sliding path.
#[test]
fn bounded_stateless_all_valid_mask_matches_absent_mask() {
    let (model, device) = tiny_model();
    let model = model
        .with_attention_execution_policy(AttentionExecutionPolicy::query_chunked(2, 20))
        .expect("two-row chunks");
    let options = ForwardOptions {
        logits: LogitsSelection::All,
        ..Default::default()
    };
    let token_ids = || Tensor::from_ints([[1, 2, 3, 4, 5]], &device);
    let expected = model
        .forward(
            CausalLmInput {
                token_ids: token_ids(),
                token_mask: None,
                position_ids: None,
            },
            &options,
        )
        .expect("unmasked forward")
        .logits;
    let actual = model
        .forward(
            CausalLmInput {
                token_ids: token_ids(),
                token_mask: Some(TokenMask::new(Tensor::from_bool(
                    TensorData::from([[true, true, true, true, true]]),
                    &device,
                ))),
                position_ids: None,
            },
            &options,
        )
        .expect("all-valid forward")
        .logits;
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(2e-6));
}

/// Muse initialization selects the two inverse-root graphs used upstream.
#[test]
fn model_selects_pinned_inverse_root_modes() {
    let (model, _) = tiny_model();
    assert_eq!(model.model.norm.root_mode, RmsNormRootMode::MultiplyByPower);
    for layer in &model.model.layers {
        for mode in [
            layer.input_layernorm.root_mode,
            layer.post_attention_layernorm.root_mode,
            layer.pre_feedforward_layernorm.root_mode,
            layer.post_feedforward_layernorm.root_mode,
        ] {
            assert_eq!(mode, RmsNormRootMode::MultiplyByReciprocalSqrt);
        }
    }
}

/// Scale-free Muse norms multiply by `pow(mean_square, -0.5)` before F16 rounding.
#[test]
fn scale_free_norm_matches_pinned_f16_power_oracle() {
    let device = Default::default();
    let input = Tensor::<TestBackend, 2>::from_floats([[0.096_557_62, 2.664_062_5]], &device)
        .cast(DType::F16);
    let actual = scale_free_rms_norm(input.clone(), 1e-5)
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();
    let fp32 = input.cast(DType::F32);
    let rms = (fp32.clone().square().mean_dim(1) + 1e-5).sqrt();
    let division = (fp32 / rms)
        .cast(DType::F16)
        .into_data()
        .to_vec::<f16>()
        .unwrap()
        .into_iter()
        .map(f16::to_bits)
        .collect::<Vec<_>>();

    assert_eq!(actual, [0x2a8f, 0x3da7]);
    assert_eq!(division, [0x2a8e, 0x3da7]);
}

/// Sliding masks measure an interior hole in semantic rather than physical time.
#[test]
fn sliding_mask_uses_valid_token_ordinals() {
    let device = Default::default();
    let valid = TokenMask::new(Tensor::<TestBackend, 2, Bool>::from_bool(
        [[true, true, false, true, true]].into(),
        &device,
    ));
    let excluded = stateless_attention_mask(Some(&valid), 1, 1, 5, Some(3), &device)
        .unwrap()
        .unwrap()
        .reshape([5, 5])
        .into_data()
        .to_vec::<bool>()
        .unwrap();

    assert_eq!(
        excluded,
        [
            false, false, true, false, false, // semantic query 0
            false, false, true, false, false, // semantic query 1
            true, true, true, true, true, // invalid query column
            false, false, true, false, false, // semantic query 2
            true, false, true, false, false, // semantic query 3
        ]
    );
}

/// Stateless inference emits exact requested shapes and dense diagnostics.
#[test]
fn stateless_forward_shapes_and_options_are_exact() {
    let (model, device) = tiny_model();
    let output = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<TestBackend, 2, Int>::from_ints(
                    [[1, 2, 3], [4, 5, 6]],
                    &device,
                ),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions {
                logits: LogitsSelection::All,
                hidden_states: true,
                ..Default::default()
            },
        )
        .expect("tiny forward");
    assert_eq!(output.logits.dims(), [2, 3, 32]);
    assert_eq!(output.hidden_states.len(), 6);
    assert_eq!(output.final_hidden_state.unwrap().dims(), [2, 3, 8]);
    assert!(output.attentions.is_empty());
    assert!(output.routers.is_empty());
}

/// NoPE makes a global layer invariant to otherwise different position IDs.
#[test]
fn nope_layer_does_not_consume_positions() {
    let (model, device) = tiny_model();
    let layer = &model.model.layers[3];
    assert!(!layer.uses_rope());
    let hidden = Tensor::<TestBackend, 3>::from_floats(
        [[[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8]]],
        &device,
    );
    let first = layer
        .forward(
            hidden.clone(),
            Tensor::from_ints([[0]], &device),
            None,
            1,
            None,
        )
        .unwrap();
    let later = layer
        .forward(hidden, Tensor::from_ints([[17]], &device), None, 18, None)
        .unwrap();
    first
        .into_data()
        .assert_approx_eq::<f32>(&later.into_data(), Tolerance::absolute(1e-6));
}

/// The separate gate is sigmoid-activated before the output projection.
#[test]
fn attention_gate_precedes_output_projection() {
    let device = Default::default();
    let config = MuseGlimmerTextConfig {
        hidden_size: 2,
        intermediate_size: 4,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        max_position_embeddings: 8,
        sliding_window: Some(2),
        layer_types: vec![MuseGlimmerLayerType::FullAttention],
        layer_rope_theta: vec![0.0],
        vocab_size: 8,
        bos_token_id: None,
        eos_token_id: None,
        ..Default::default()
    };
    let mut attention = MuseGlimmerAttention::<TestBackend>::init(&config, &device).unwrap();
    let identity = Tensor::from_floats([[1.0, 0.0], [0.0, 1.0]], &device);
    attention.q_proj.weight = Param::from_tensor(identity.clone());
    attention.k_proj.weight = Param::from_tensor(identity.clone());
    attention.v_proj.weight = Param::from_tensor(identity.clone());
    attention.o_proj.weight = Param::from_tensor(identity);
    attention.gate_proj.weight = Param::from_tensor(Tensor::zeros([2, 2], &device));
    let hidden = Tensor::from_floats([[[3.0, 4.0]]], &device);
    let projection = attention.project_self(hidden).unwrap();
    let value = projection.value.clone().swap_dims(1, 2).reshape([1, 1, 2]);
    let actual = attention.attend(projection, None, true).unwrap();
    (actual * 2.0)
        .into_data()
        .assert_approx_eq::<f32>(&value.into_data(), Tolerance::absolute(1e-6));
}

/// Query heads are scale-free normalized and then multiplied by the configured factor.
#[test]
fn query_normalization_applies_extra_scale() {
    let (model, device) = tiny_model();
    let attention = &model.model.layers[0].self_attn;
    let hidden = Tensor::<TestBackend, 3>::ones([1, 1, 8], &device);
    let projection = attention.project_self(hidden).unwrap();
    let query_rms = projection
        .query
        .square()
        .mean_dim(3)
        .sqrt()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    for value in query_rms {
        assert!((value - attention.qk_scale_factor as f32).abs() < 2e-3);
    }
}

/// The two residual branches follow the documented four-norm sandwich order.
#[test]
fn decoder_layer_matches_explicit_sandwich_equation() {
    let (model, device) = tiny_model();
    let layer = &model.model.layers[3];
    let hidden = Tensor::<TestBackend, 3>::from_floats(
        [[[0.2, -0.1, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8]]],
        &device,
    );
    let position = Tensor::from_ints([[0]], &device);
    let normalized = layer.input_layernorm.forward(hidden.clone());
    let projection = layer.self_attn.project_self(normalized).unwrap();
    let attended = layer.self_attn.attend(projection, None, true).unwrap();
    let after_attention = hidden.clone() + layer.post_attention_layernorm.forward(attended);
    let fed = layer
        .mlp
        .forward(
            layer
                .pre_feedforward_layernorm
                .forward(after_attention.clone()),
        )
        .expect("validated feed-forward projection");
    let expected = after_attention + layer.post_feedforward_layernorm.forward(fed);
    let actual = layer.forward(hidden, position, None, 1, Some(3)).unwrap();
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
}

/// Final logits match the exact output multiplier and tanh softcap equation.
#[test]
fn final_logit_softcap_matches_closed_form_and_bounds() {
    let (model, device) = tiny_model();
    let hidden = Tensor::<TestBackend, 3>::from_floats(
        [[[0.5, -1.0, 1.5, 0.25, -0.75, 2.0, 0.0, -0.5]]],
        &device,
    );
    let raw = linear(hidden.clone(), model.lm_head.as_ref().unwrap()).unwrap();
    let expected =
        ((raw * model.config().output_multiplier) / model.config().final_logit_softcapping).tanh()
            * model.config().final_logit_softcapping;
    let actual = model.project_logits(hidden).unwrap();
    actual
        .clone()
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::absolute(1e-6));
    assert!(
        actual.abs().max().into_scalar().elem::<f32>()
            <= model.config().final_logit_softcapping as f32
    );
}

/// Invalid input values and unsupported diagnostics return contextual errors.
#[test]
fn public_forward_rejects_invalid_inputs_without_panicking() {
    let (model, device) = tiny_model();
    let invalid_token = model.forward(
        CausalLmInput {
            token_ids: Tensor::from_ints([[32]], &device),
            token_mask: None,
            position_ids: None,
        },
        &ForwardOptions::default(),
    );
    assert!(invalid_token.is_err());
    let attention_request = model.forward(
        CausalLmInput {
            token_ids: Tensor::from_ints([[1]], &device),
            token_mask: None,
            position_ids: None,
        },
        &ForwardOptions {
            attentions: true,
            ..Default::default()
        },
    );
    assert!(matches!(
        attention_request,
        Err(Error::BackendCapability(_))
    ));
}

/// Stateless device preflight identifies every independently supplied tensor.
#[test]
fn stateless_device_affinity_errors_are_contextual() {
    assert!(ensure_same_device("muse_glimmer.input_ids", &0_u8, &0_u8, PARAMETERS).is_ok());
    for tensor in [
        "muse_glimmer.input_ids",
        "muse_glimmer.token_mask",
        "muse_glimmer.position_ids",
        "muse_glimmer.lm_head",
    ] {
        let error = ensure_same_device(tensor, &1_u8, &0_u8, PARAMETERS)
            .expect_err("different device identities must fail preflight");
        assert!(error.to_string().contains(tensor));
        assert!(error.to_string().contains("model parameters"));
    }
}

/// Stateless graph construction remains differentiable through embeddings and logits.
#[cfg(feature = "autodiff")]
#[test]
fn autodiff_smoke_reaches_embedding_weights() {
    use burn::backend::Autodiff;

    /// Reverse-mode backend for checking gradient propagation through Muse attention.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let model = MuseGlimmerForCausalLm::<AdBackend>::init(tiny_config(), &device).unwrap();
    let loss = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .unwrap()
        .logits
        .sum();
    let gradients = loss.backward();
    assert!(model.model.embed_tokens.weight.grad(&gradients).is_some());
}

/// Multi-chunk autodiff fails closed because its tape is not memory-bounded.
#[cfg(feature = "autodiff")]
#[test]
fn bounded_multichunk_autodiff_is_not_mislabeled_as_memory_bounded() {
    use burn::backend::Autodiff;

    /// Reverse-mode backend for checking gradient propagation through Muse attention.
    type AdBackend = Autodiff<TestBackend>;
    let device = Default::default();
    let model = MuseGlimmerForCausalLm::<AdBackend>::init(tiny_config(), &device)
        .unwrap()
        .with_attention_execution_policy(AttentionExecutionPolicy::query_chunked(1, 64))
        .unwrap();
    let error = model
        .forward(
            CausalLmInput {
                token_ids: Tensor::<AdBackend, 2, Int>::from_ints([[1, 2]], &device),
                token_mask: None,
                position_ids: None,
            },
            &ForwardOptions::default(),
        )
        .expect_err("autodiff retains all chunk intermediates");
    assert!(error.to_string().contains("inference-only"));
    assert!(error.to_string().contains("autodiff tape"));
}

impl<B: Backend> MuseGlimmerAttention<B> {
    /// Evaluates already positioned attention and gates its result before `o_proj`.
    ///
    /// `mask` follows Burn's kernel convention: `true` excludes a query/key pair.
    /// This entry point can slice score computation but cannot undo an already
    /// materialized quadratic mask. Stateless and cached model paths use the
    /// lazy-mask entry points below for long contexts.
    #[cfg(test)]
    fn attend(
        &self,
        projection: MuseAttentionProjection<B>,
        mask: Option<Tensor<B, 4, Bool>>,
        causal: bool,
    ) -> Result<Tensor<B, 3>> {
        let [batch, query_sequence, key_sequence] = self.validate_projection(&projection)?;
        validate_materialized_mask(
            mask.as_ref(),
            batch,
            self.query_heads,
            query_sequence,
            key_sequence,
        )?;
        let plan =
            self.execution_policy
                .plan(batch, self.query_heads, query_sequence, key_sequence)?;
        if plan.chunk_count() == 1 {
            return self.attend_unbounded(projection, mask, causal);
        }
        self.attend_chunked(projection, plan, causal, None, |query_range, key_range| {
            Ok(mask.as_ref().map(|mask| {
                mask.clone()
                    .slice([0..batch, 0..self.query_heads, query_range, key_range])
            }))
        })
    }
}
