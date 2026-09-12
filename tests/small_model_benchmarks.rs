//! Reproducible, opt-in FLAN-T5-small F32 benchmarks on Flex and Metal.
//!
//! Run the separate official-reference acceptance test before publishing these samples. This
//! harness additionally authenticates the checkpoint and reference bundle, checks complete
//! first/cached reference logits and 32 greedy decisions, then times three warmups and ten
//! repetitions for 32-, 128- and 512-token sources with 128 forced cached decode steps.
//!
//! The environment variables `TRANSFORMERS_FLAN_T5_SMALL_DIR` and
//! `TRANSFORMERS_FLAN_T5_SMALL_REFERENCE_DIR` select existing local artifacts.
//! `TRANSFORMERS_BENCH_METADATA_JSON` points to a JSON `RunEnvironment` record (see the shared
//! metrics module) containing exact machine/runtime/power/exclusivity/command facts.
//! `TRANSFORMERS_BENCH_OUTPUT_JSON` optionally selects a create-only report destination; without
//! it the compact report is printed after `BENCHMARK_JSON=`. Run one ignored test in a fresh
//! release-profile process with `--test-threads=1`. For example:
//!
//! ```text
//! /usr/bin/time -l cargo test --release --no-default-features --features std,metal,tokenizers \
//!   --test small_model_benchmarks flan::flan_small_metal_benchmark \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```
//!
//! Timing uses `Backend::sync` on both sides of every phase. Model-only decode excludes host
//! argmax/readback; sampling-inclusive decode includes the same forced sequence plus greedy
//! selection/readback and loop overhead. EOS cannot shorten that 128-step denominator. A separate
//! natural greedy translation uses normal EOS handling and has its own variable token count.
//! RSS observations are explicitly boundary snapshots, never peaks. True process/unified
//! high-water evidence must come from a separate supervisor such as `/usr/bin/time -l` applied
//! to the already-built test executable (wrapping Cargo also includes the build).

#[path = "support/benchmark_metrics.rs"]
mod benchmark_metrics;

#[cfg(all(
    feature = "tokenizers",
    any(
        feature = "flex",
        all(feature = "metal", target_os = "macos", not(feature = "rocm"))
    )
))]
mod flan {
    //! Authenticated FLAN-T5-small benchmark runner with explicit timing and memory boundaries.

    use std::{
        collections::BTreeSet,
        fs,
        io::{Read, Write},
        path::{Path, PathBuf},
        process::Command,
        time::Instant,
    };

    use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use transformers::{
        AutoLoadOptions, AutoSeq2SeqLm, CausalLmInput, DecodeInput, DecodeState, EncodeOptions,
        ForwardOptions, LogitsSelection, Seq2SeqLanguageModel,
        cache::CacheConfig,
        models::t5::{
            T5BeamSearchOptions, T5DecodeState, T5ForConditionalGeneration, generate_t5_beam_search,
        },
    };

    use super::benchmark_metrics::{
        DECODE_STEPS, MEASURED_REPETITIONS, MemoryPhase, RunEnvironment, WARMUP_REPETITIONS,
        memory_profiling_enabled, summarize, t5_cache_accounting, token_hash,
    };

    /// Pinned official artifact identity shared with the reference acceptance record.
    const REPOSITORY: &str = "google/flan-t5-small";
    /// Immutable checkpoint revision.
    const REVISION: &str = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab";
    /// Accepted evidence also serves as the authoritative seven-file hash allowlist.
    const ACCEPTANCE: &str =
        include_str!("../acceptance/flan-t5-small-0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab.json");
    /// Source used for normal-EOS greedy generation, separate from synthetic length fixtures.
    const TRANSLATION_PROMPT: &str = "translate English to German: The house is wonderful.";

    /// Returns a required path argument without consulting the network or a global model cache.
    fn required_path(name: &str) -> PathBuf {
        std::env::var_os(name)
            .map(PathBuf::from)
            .unwrap_or_else(|| panic!("set {name} before running the benchmark"))
    }

    /// Streams an artifact hash without staging the full checkpoint in host memory.
    fn file_hash(path: &Path) -> String {
        let mut file =
            fs::File::open(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer).expect("read artifact for SHA-256");
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
        format!("{:x}", digest.finalize())
    }

    /// Authenticates the exact artifact set using the committed acceptance hash table.
    ///
    /// The original FLAN integration test's helpers are private. Reading its committed record
    /// here avoids copying a second mutable list of expected hashes or weakening authentication.
    fn authenticate_checkpoint(root: &Path, accepted: &Value) -> PathBuf {
        let root = root.canonicalize().expect("canonical checkpoint root");
        let expected = accepted["sha256"]
            .as_object()
            .expect("committed artifact hash inventory");
        assert_eq!(expected.len(), 7);
        let mut observed = BTreeSet::new();
        for entry in fs::read_dir(&root).expect("list checkpoint artifact root") {
            let entry = entry.unwrap();
            let name = entry
                .file_name()
                .into_string()
                .expect("UTF-8 artifact basename");
            let metadata = fs::symlink_metadata(entry.path()).unwrap();
            assert!(
                !metadata.file_type().is_symlink(),
                "symlink artifact {name}"
            );
            if name == ".cache" {
                assert!(metadata.is_dir());
                continue;
            }
            assert!(metadata.is_file(), "unexpected artifact directory {name}");
            observed.insert(name);
        }
        assert_eq!(observed, expected.keys().cloned().collect());
        for (name, expected_hash) in expected {
            assert_eq!(
                file_hash(&root.join(name)),
                expected_hash.as_str().unwrap(),
                "artifact {name}"
            );
        }
        root
    }

    /// Reads one authenticated reference vector under a contained, fixed bundle basename.
    fn reference_vector(root: &Path, reference: &Value, key: &str) -> Vec<f32> {
        let descriptor = &reference["tensors"][key];
        let name = descriptor["relative_path"]
            .as_str()
            .expect("reference filename");
        assert_eq!(Path::new(name).components().count(), 1);
        assert!(!Path::new(name).is_absolute());
        let path = root.join(name);
        let metadata = fs::symlink_metadata(&path).unwrap();
        assert!(metadata.is_file() && !metadata.file_type().is_symlink());
        assert_eq!(metadata.len(), descriptor["bytes"].as_u64().unwrap());
        assert!(metadata.len() <= 5 * 1024 * 1024);
        assert_eq!(file_hash(&path), descriptor["sha256"].as_str().unwrap());
        let raw = fs::read(path).unwrap();
        assert!(raw.len().is_multiple_of(4));
        raw.chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    /// Checks complete F32 vectors against the established elementwise acceptance tolerance.
    fn assert_reference(actual: &[f32], reference: &[f32]) {
        assert_eq!(actual.len(), reference.len());
        for (index, (&actual, &reference)) in actual.iter().zip(reference).enumerate() {
            assert!(actual.is_finite() && reference.is_finite());
            assert!(
                (actual - reference).abs() <= 1.0e-4 + 1.0e-4 * reference.abs(),
                "reference mismatch at {index}: {actual} vs {reference}"
            );
        }
    }

    /// Executes a small metadata command and preserves failure as evidence instead of guessing.
    fn command_text(program: &str, arguments: &[&str], directory: &Path) -> Value {
        match Command::new(program)
            .args(arguments)
            .current_dir(directory)
            .output()
        {
            Ok(output) if output.status.success() => {
                json!({"value": String::from_utf8_lossy(&output.stdout).trim(), "error": null})
            }
            Ok(output) => {
                json!({"value": null, "error": String::from_utf8_lossy(&output.stderr).trim()})
            }
            Err(error) => json!({"value": null, "error": error.to_string()}),
        }
    }

    /// Captures an explicitly labeled process-RSS boundary observation, not a memory high-water.
    fn rss_snapshot(phase: &str) -> Value {
        let pid = std::process::id().to_string();
        let result = command_text(
            "ps",
            &["-o", "rss=", "-p", &pid],
            Path::new(env!("CARGO_MANIFEST_DIR")),
        );
        let bytes = result["value"]
            .as_str()
            .and_then(|text| text.parse::<u64>().ok())
            .and_then(|kib| kib.checked_mul(1024));
        let error = if bytes.is_none() && result["error"].is_null() {
            json!("RSS output could not be parsed as a KiB count")
        } else {
            result["error"].clone()
        };
        json!({"phase": phase, "resident_bytes": bytes, "measurement": "ps RSS boundary snapshot", "is_peak": false, "error": error})
    }

    /// Completes backend and fusion work before a wall-time boundary is observed.
    fn synchronize<B: Backend>(device: &B::Device) {
        B::sync(device).expect("benchmark backend synchronization");
    }

    /// Constructs a batch-one valid source or target tensor outside timed inference.
    fn input<B: Backend>(ids: &[u32], device: &B::Device) -> CausalLmInput<B> {
        CausalLmInput {
            token_ids: Tensor::<B, 2, Int>::from_data(
                TensorData::new(
                    ids.iter().map(|&id| i64::from(id)).collect::<Vec<_>>(),
                    [1, ids.len()],
                ),
                device,
            ),
            token_mask: None,
            position_ids: None,
        }
    }

    /// Greedy selection transfers one token ID, not the full vocabulary, to the host.
    fn select_token<B: Backend>(logits: Tensor<B, 3>) -> u32 {
        let data = logits.argmax(2).reshape([1]).into_data().convert::<i64>();
        u32::try_from(data.to_vec::<i64>().unwrap()[0]).expect("greedy token is a u32 ID")
    }

    /// Extracts F32 data only for correctness validation outside throughput intervals.
    fn values<B: Backend>(logits: Tensor<B, 3>) -> Vec<f32> {
        logits
            .into_data()
            .to_vec::<f32>()
            .expect("F32 benchmark backend")
    }

    /// Creates the exact decoder cache capacity needed for the start token plus forced steps.
    fn new_state<B: Backend>(
        model: &T5ForConditionalGeneration<B>,
        device: &B::Device,
    ) -> T5DecodeState<B> {
        model
            .new_decode_state(CacheConfig::new(1, DECODE_STEPS + 1).unwrap(), 1, device)
            .unwrap()
    }

    /// Counts actual exposed key/value tensor payloads, without counting allocator reservation.
    fn observed_self_cache_bytes<B: Backend>(state: &T5DecodeState<B>) -> u64 {
        let kv = state.tensor_state().main().native_kv(0).unwrap();
        let mut bytes = 0_u64;
        for layer in 0..kv.spec().layers() {
            for tensor in [kv.keys(layer).unwrap(), kv.values(layer).unwrap()] {
                assert_eq!(tensor.dtype(), burn::tensor::DType::F32);
                bytes += tensor.dims().into_iter().product::<usize>() as u64 * 4;
            }
        }
        bytes
    }

    /// One fixed-work sample with paired model-only and host-selection-inclusive decode times.
    struct Sample {
        /// Encoder and decoder-start prefill wall time, excluding token selection.
        prefill_seconds: f64,
        /// Same prefill through first greedy token readback.
        first_token_seconds: f64,
        /// Sum of synchronized model-only cached step intervals.
        decode_model_seconds: f64,
        /// Forced decode loop including per-step greedy selection and host bookkeeping.
        decode_inclusive_seconds: f64,
        /// Full greedy predictions at every forced prefix; these are not fed back as inputs.
        predictions: Vec<u32>,
    }

    /// Executes fresh-state prefill and 128 teacher-forced cached steps with fixed valid counts.
    fn sample<B: Backend>(
        model: &T5ForConditionalGeneration<B>,
        source: CausalLmInput<B>,
        forced: &[CausalLmInput<B>],
        device: &B::Device,
    ) -> Sample {
        let mut state = new_state(model, device);
        let options = ForwardOptions {
            logits: LogitsSelection::Last,
            ..ForwardOptions::default()
        };
        let start_token = input::<B>(&[0], device);
        synchronize::<B>(device);
        let source_length = source.token_ids.dims()[1];
        let prefill_phase = MemoryPhase::begin(&format!("warm_source_{source_length}_prefill"));
        let start = Instant::now();
        let encoder = model.encode(source, &ForwardOptions::default()).unwrap();
        let prefill = model
            .prefill_decoder(&encoder, start_token, &mut state, &options)
            .unwrap();
        synchronize::<B>(device);
        let prefill_seconds = start.elapsed().as_secs_f64();
        let first = select_token(prefill.decoder.logits);
        synchronize::<B>(device);
        let first_token_seconds = start.elapsed().as_secs_f64();
        prefill_phase.finish();
        let mut predictions = vec![first];
        let mut decode_model_seconds = 0.0;
        let decode_phase = MemoryPhase::begin(&format!("warm_source_{source_length}_decode_128"));
        let start = Instant::now();
        for token in forced {
            synchronize::<B>(device);
            let step = Instant::now();
            let output = model
                .decode(
                    &encoder,
                    DecodeInput {
                        token_ids: token.token_ids.clone(),
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &options,
                )
                .unwrap();
            synchronize::<B>(device);
            decode_model_seconds += step.elapsed().as_secs_f64();
            predictions.push(select_token(output.decoder.logits));
        }
        synchronize::<B>(device);
        let decode_inclusive_seconds = start.elapsed().as_secs_f64();
        decode_phase.finish();
        assert_eq!(state.lengths(), [DECODE_STEPS + 1]);
        assert_eq!(predictions.len(), DECODE_STEPS + 1);
        assert!(first_token_seconds >= prefill_seconds);
        assert!(decode_inclusive_seconds >= decode_model_seconds);
        Sample {
            prefill_seconds,
            first_token_seconds,
            decode_model_seconds,
            decode_inclusive_seconds,
            predictions,
        }
    }

    /// Measures cold first-token latency, then validates its full vector and all cached decisions.
    fn cold_and_reference<B: Backend>(
        model: &T5ForConditionalGeneration<B>,
        source: CausalLmInput<B>,
        reference: &Value,
        reference_root: &Path,
        device: &B::Device,
    ) -> Value {
        let expected_first = reference_vector(reference_root, reference, "first_logits");
        let expected_cached = reference_vector(reference_root, reference, "cached_logits");
        let expected_ids: Vec<u32> =
            serde_json::from_value(reference["generation"]["token_ids"].clone()).unwrap();
        assert_eq!(expected_ids.len(), 32);
        let mut state = new_state(model, device);
        let options = ForwardOptions {
            logits: LogitsSelection::Last,
            ..ForwardOptions::default()
        };
        let start_token = input::<B>(&[0], device);
        synchronize::<B>(device);
        let phase = MemoryPhase::begin("cold_prefill");
        let start = Instant::now();
        let encoder = model.encode(source, &ForwardOptions::default()).unwrap();
        let output = model
            .prefill_decoder(&encoder, start_token, &mut state, &options)
            .unwrap();
        let first = select_token(output.decoder.logits.clone());
        synchronize::<B>(device);
        let seconds = start.elapsed().as_secs_f64();
        phase.finish();
        let first_values = values(output.decoder.logits);
        assert_reference(&first_values, &expected_first);
        assert_reference(&first_values, &expected_cached[..model.config().vocab_size]);
        assert_eq!(first, expected_ids[0]);
        for decision in 1..32 {
            let token = input::<B>(&[expected_ids[decision - 1]], device);
            let output = model
                .decode(
                    &encoder,
                    DecodeInput {
                        token_ids: token.token_ids,
                        token_mask: None,
                        position_ids: None,
                    },
                    &mut state,
                    &options,
                )
                .unwrap();
            assert_eq!(
                select_token(output.decoder.logits.clone()),
                expected_ids[decision]
            );
            let start = decision * model.config().vocab_size;
            assert_reference(
                &values(output.decoder.logits),
                &expected_cached[start..start + model.config().vocab_size],
            );
        }
        json!({"seconds": seconds, "shape": "batch1,51 source tokens,1 decoder start token", "includes_cache_allocation": false, "includes_token_selection_and_readback": true, "includes_lazy_kernel_compilation": true, "reference_cached_decisions": 32, "token_ids": expected_ids, "token_ids_u32le_sha256": token_hash(&expected_ids)})
    }

    /// Allows debug compilation of ignored tests while rejecting accidental debug measurements.
    #[allow(
        clippy::assertions_on_constants,
        reason = "ignored benchmarks compile in debug test suites but execute only in release"
    )]
    fn require_release() {
        assert!(!cfg!(debug_assertions), "benchmark requires --release");
    }

    /// Runs one homogeneous backend/dtype benchmark series and emits a compact reproducible record.
    fn run<B: Backend>(device: &B::Device, backend_profile: &str) {
        require_release();
        let metadata_bytes = fs::read(required_path("TRANSFORMERS_BENCH_METADATA_JSON")).unwrap();
        let environment: RunEnvironment =
            serde_json::from_slice(&metadata_bytes).expect("complete benchmark environment JSON");
        environment.validate().unwrap();
        if backend_profile == "Metal/F32" {
            assert!(
                environment.accelerator_exclusive,
                "reserve the Metal device before benchmarking"
            );
        }
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let benchmark_source_sha256 = format!(
            "{:x}",
            Sha256::digest(include_bytes!("small_model_benchmarks.rs"))
        );
        let metric_helper_sha256 = format!(
            "{:x}",
            Sha256::digest(include_bytes!("support/benchmark_metrics.rs"))
        );
        assert_eq!(
            file_hash(&manifest_dir.join("tests/small_model_benchmarks.rs")),
            benchmark_source_sha256,
            "benchmark source changed after compilation"
        );
        assert_eq!(
            file_hash(&manifest_dir.join("tests/support/benchmark_metrics.rs")),
            metric_helper_sha256,
            "metric helper changed after compilation"
        );
        let accepted: Value = serde_json::from_str(ACCEPTANCE).unwrap();
        assert_eq!(accepted["repository"], REPOSITORY);
        assert_eq!(accepted["revision"], REVISION);
        let auth_started = Instant::now();
        let root =
            authenticate_checkpoint(&required_path("TRANSFORMERS_FLAN_T5_SMALL_DIR"), &accepted);
        let authentication_seconds = auth_started.elapsed().as_secs_f64();
        let reference_root = required_path("TRANSFORMERS_FLAN_T5_SMALL_REFERENCE_DIR")
            .canonicalize()
            .unwrap();
        let reference_path = reference_root.join("flan-t5-small-0fc9ddf7-reference.json");
        assert_eq!(
            file_hash(&reference_path),
            accepted["reference_acceptance"]["oracle"]["bundle_manifest_sha256"]
                .as_str()
                .unwrap()
        );
        let reference: Value = serde_json::from_slice(&fs::read(reference_path).unwrap()).unwrap();
        let mut memory = vec![rss_snapshot("before_load")];
        synchronize::<B>(device);
        let phase = MemoryPhase::begin("strict_load");
        let start = Instant::now();
        let loaded = AutoSeq2SeqLm::<B>::load_local(
            &root,
            &AutoLoadOptions {
                repository: Some(REPOSITORY.into()),
                revision: Some(REVISION.into()),
                ..AutoLoadOptions::default()
            },
            device,
        )
        .unwrap();
        synchronize::<B>(device);
        let load_seconds = start.elapsed().as_secs_f64();
        phase.finish();
        let report = loaded.load_report();
        assert_eq!(report.applied.len(), 190);
        assert_eq!(report.transformed.len(), 145);
        assert!(report.missing.is_empty() && report.unexpected.is_empty());
        assert!(report.quantized.is_empty() && report.dtype_converted.is_empty());
        assert!(report.allowlisted_missing.is_empty() && report.allowlisted_ignored.is_empty());
        memory.push(rss_snapshot("after_load"));
        let model = match loaded.model() {
            AutoSeq2SeqLm::T5(model) => model,
            #[allow(unreachable_patterns)]
            _ => panic!("wrong automatic model family"),
        };
        let tokenizer = loaded.tokenizer().expect("strict local tokenizer load");
        let prompt = reference["generation"]["prompt"].as_str().unwrap();
        let source = tokenizer.encode(prompt, &EncodeOptions::default()).unwrap();
        let source_ids: Vec<u32> =
            serde_json::from_value(reference["generation"]["source_token_ids"].clone()).unwrap();
        assert_eq!(source.token_ids(), &[source_ids]);
        let cold = cold_and_reference(
            model,
            source.to_causal_input::<B>(device).unwrap(),
            &reference,
            &reference_root,
            device,
        );
        memory.push(rss_snapshot("after_cold_and_reference"));
        let forced_ids = (0..DECODE_STEPS)
            .map(|index| [71_u32, 8, 57][index % 3])
            .collect::<Vec<_>>();
        let forced = forced_ids
            .iter()
            .map(|&id| input::<B>(&[id], device))
            .collect::<Vec<_>>();
        let mut series = Vec::new();
        for source_length in [32_usize, 128, 512] {
            let source_ids = (0..source_length - 1)
                .map(|index| [71_u32, 8, 57][index % 3])
                .chain([1])
                .collect::<Vec<_>>();
            let source = input::<B>(&source_ids, device);
            for _ in 0..WARMUP_REPETITIONS {
                std::hint::black_box(sample(model, source.clone(), &forced, device));
            }
            let mut samples = Vec::new();
            for _ in 0..MEASURED_REPETITIONS {
                samples.push(sample(model, source.clone(), &forced, device));
            }
            for sample in &samples {
                assert_eq!(
                    sample.predictions, samples[0].predictions,
                    "warm repeated predictions drifted"
                );
            }
            let config = model.config();
            let accounting = t5_cache_accounting(
                config.decoder_layers() as u64,
                config.num_heads as u64,
                config.d_kv as u64,
                config.d_model as u64,
                4,
                1,
                (DECODE_STEPS + 1) as u64,
                source_length as u64,
            )
            .unwrap();
            let state = new_state(model, device);
            let observed_self = observed_self_cache_bytes(&state);
            assert_eq!(accounting.self_kv_allocated_bytes, observed_self);
            series.push(json!({
                "source_length": source_length, "batch_size":1, "valid_source_tokens":source_length, "padding":"none",
                "source_token_ids":source_ids, "source_tokens_u32le_sha256":token_hash(&source_ids),
                "target_cache_capacity":DECODE_STEPS+1, "cache_window":null, "cache":accounting,
                "observed_self_kv_tensor_payload_bytes":observed_self,
                "cross_kv_accounting":"analytical; private immutable cache storage is not publicly inspectable",
                "encoder_decoder_prefill":summarize(&samples.iter().map(|s|s.prefill_seconds).collect::<Vec<_>>(),Some(source_length),MEASURED_REPETITIONS).unwrap(),
                "first_token_latency":summarize(&samples.iter().map(|s|s.first_token_seconds).collect::<Vec<_>>(),None,MEASURED_REPETITIONS).unwrap(),
                "decode_model_only":summarize(&samples.iter().map(|s|s.decode_model_seconds).collect::<Vec<_>>(),Some(DECODE_STEPS),MEASURED_REPETITIONS).unwrap(),
                "decode_including_greedy_readback":summarize(&samples.iter().map(|s|s.decode_inclusive_seconds).collect::<Vec<_>>(),Some(DECODE_STEPS),MEASURED_REPETITIONS).unwrap(),
                "predicted_token_ids":samples[0].predictions, "predicted_tokens_u32le_sha256":token_hash(&samples[0].predictions),
                "per_repetition_prediction_hashes":samples.iter().map(|s|token_hash(&s.predictions)).collect::<Vec<_>>()
            }));
            memory.push(rss_snapshot(&format!("after_source_{source_length}")));
        }
        let translation = tokenizer
            .encode(TRANSLATION_PROMPT, &EncodeOptions::default())
            .unwrap();
        let translation_ids = translation.token_ids()[0].clone();
        let translation = translation.to_causal_input::<B>(device).unwrap();
        let mut natural_samples = Vec::new();
        let mut natural_tokens: Option<Vec<u32>> = None;
        let mut natural_finish = String::new();
        for repetition in 0..WARMUP_REPETITIONS + MEASURED_REPETITIONS {
            synchronize::<B>(device);
            let phase = MemoryPhase::begin("natural_greedy");
            let start = Instant::now();
            let output = generate_t5_beam_search(
                model,
                translation.clone(),
                T5BeamSearchOptions {
                    num_beams: 1,
                    max_new_tokens: 64,
                    ..T5BeamSearchOptions::default()
                },
            )
            .unwrap();
            synchronize::<B>(device);
            let seconds = start.elapsed().as_secs_f64();
            phase.finish();
            let sequence = &output.sequences[0][0];
            if let Some(expected) = &natural_tokens {
                assert_eq!(&sequence.token_ids, expected);
            }
            natural_tokens = Some(sequence.token_ids.clone());
            natural_finish = format!("{:?}", sequence.finish_reason);
            if repetition >= WARMUP_REPETITIONS {
                natural_samples.push(seconds);
            }
        }
        let natural_tokens = natural_tokens.unwrap();
        memory.push(rss_snapshot("after_natural_greedy"));
        let git_commit = command_text("git", &["rev-parse", "HEAD"], manifest_dir);
        let git_status = command_text(
            "git",
            &["status", "--porcelain", "--untracked-files=all", "--", "."],
            manifest_dir,
        );
        let natural = json!({
            "prompt":TRANSLATION_PROMPT,"source_token_ids":translation_ids,"source_tokens_u32le_sha256":token_hash(&translation_ids),
            "num_beams":1,"eos_token_ids":[1],"pad_token_id":0,"max_new_tokens":64,"cache_allocation_included":true,
            "token_ids":natural_tokens,"tokens_u32le_sha256":token_hash(&natural_tokens),"finish_reason":natural_finish,
            "total_time":summarize(&natural_samples,None,MEASURED_REPETITIONS).unwrap()
        });
        let compiled_toolchain = option_env!("RUSTUP_TOOLCHAIN");
        let compiler = match compiled_toolchain {
            Some(toolchain) => command_text(
                "rustc",
                &[&format!("+{toolchain}"), "--version", "--verbose"],
                manifest_dir,
            ),
            None => {
                json!({"value":null,"error":"RUSTUP_TOOLCHAIN was not present when this executable was compiled"})
            }
        };
        let provenance = json!({
            "repository":REPOSITORY, "revision":REVISION,
            "artifact_sha256":accepted["sha256"], "reference_bundle_sha256":accepted["reference_acceptance"]["oracle"]["bundle_manifest_sha256"],
            "crate_commit":git_commit, "crate_worktree_status":git_status, "rustc":compiler,"compiled_rustup_toolchain":compiled_toolchain,
            "cargo_lock_sha256":file_hash(&manifest_dir.join("Cargo.lock")),
            "benchmark_source_sha256":benchmark_source_sha256,
            "metric_helper_sha256":metric_helper_sha256,
            "benchmark_executable_sha256":file_hash(&std::env::current_exe().unwrap()),
            "burn_version":"0.21.0", "profile":"release (debug assertions disabled)", "backend_profile":backend_profile,
            "backend_name":B::name(device), "parameter_dtype":"F32", "quantization":null,
            "cargo_features":cargo_features(), "environment":environment
        });
        let protocol = json!({
            "warmup_repetitions":WARMUP_REPETITIONS, "measured_repetitions":MEASURED_REPETITIONS,
            "decode_steps":DECODE_STEPS, "decode_mode":"teacher_forced_with_greedy_predictions", "seed":0,
            "forced_decode_token_ids":forced_ids,"forced_decode_tokens_u32le_sha256":token_hash(&forced_ids),
            "synchronization":"Backend::sync before and after timed model work; readback completion included in selection-inclusive intervals",
            "tokenization_included":false,"prefill_cache_allocation_included":false
        });
        let result = json!({
            "schema_version":1,"benchmark":"flan_t5_small_f32","provenance":provenance,"protocol":protocol,
            "process_memory_instrumentation_enabled":memory_profiling_enabled(),"timing_comparable_to_uninstrumented_series":!memory_profiling_enabled(),
            "artifact_authentication_seconds_separate_from_load":authentication_seconds,
            "strict_load_seconds":load_seconds,"strict_load_report":report,"cold_first_token":cold,"warm_series":series,
            "natural_greedy":natural,
            "memory":{"boundary_snapshots":memory,"phase_peak_host_bytes":null,"phase_peak_device_bytes":null,"phase_peak_unified_bytes":null,"external_high_water_evidence":"collect separately with a supervisor around the already-built executable; boundary snapshots are not peak values","allocator_reservations_included_in_cache_payload":false},
            "publishable_without_external_memory_and_full_acceptance_evidence":false
        });
        let encoded = serde_json::to_vec(&result).unwrap();
        if let Some(path) = std::env::var_os("TRANSFORMERS_BENCH_OUTPUT_JSON") {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .expect("create-only benchmark report");
            file.write_all(&encoded).unwrap();
            file.write_all(b"\n").unwrap();
            file.sync_all().unwrap();
        } else {
            println!("BENCHMARK_JSON={}", String::from_utf8(encoded).unwrap());
        }
    }

    /// Records every runtime-affecting crate feature so incompatible series cannot be merged.
    fn cargo_features() -> Vec<&'static str> {
        [
            ("std", cfg!(feature = "std")),
            ("flex", cfg!(feature = "flex")),
            ("metal", cfg!(feature = "metal")),
            ("cpu", cfg!(feature = "cpu")),
            ("cuda", cfg!(feature = "cuda")),
            ("rocm", cfg!(feature = "rocm")),
            ("autodiff", cfg!(feature = "autodiff")),
            ("tokenizers", cfg!(feature = "tokenizers")),
            ("chat-templates", cfg!(feature = "chat-templates")),
            ("kernel-tracing", cfg!(feature = "kernel-tracing")),
            (
                "attention-kernel-tracing",
                cfg!(feature = "attention-kernel-tracing"),
            ),
        ]
        .into_iter()
        .filter_map(|(name, enabled)| enabled.then_some(name))
        .collect()
    }

    /// Runs the CPU/F32 profile with exact local checkpoint and oracle identities.
    #[cfg(feature = "flex")]
    #[test]
    #[ignore = "requires pinned local FLAN artifacts, reference bundle, environment metadata and --release"]
    fn flan_small_flex_benchmark() {
        run::<burn::backend::Flex<f32, i32>>(&Default::default(), "Flex/F32");
    }

    /// Runs the accelerator/F32 profile only in a serialized, explicitly reserved Metal session.
    #[cfg(all(feature = "metal", target_os = "macos", not(feature = "rocm")))]
    #[test]
    #[ignore = "requires pinned local FLAN artifacts, reference bundle, reserved Metal device and --release"]
    fn flan_small_metal_benchmark() {
        run::<burn::backend::Metal>(&Default::default(), "Metal/F32");
    }
}
