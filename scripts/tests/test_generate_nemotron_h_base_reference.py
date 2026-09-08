"""Pure tests for the opt-in Nemotron-H base official-reference generator."""

from __future__ import annotations

import importlib.util
import io
import json
import struct
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "generate_nemotron_h_base_reference.py"
SPEC = importlib.util.spec_from_file_location("generate_nemotron_h_base_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class FakeTorch:
    """Minimal device facade for allocator-warmup validation."""

    @staticmethod
    def device(value: object) -> object:
        """Return the leading device type from a nonempty string."""

        if not isinstance(value, str) or not value:
            raise TypeError("invalid fake device")
        return types.SimpleNamespace(type=value.partition(":")[0])


class FakeNemotronModel:
    """Exact class sentinel accepted by the warmup helper."""


def fake_modeling_utils() -> tuple[object, object]:
    """Create an upstream-shaped warmup owner and retain its callable."""

    def caching_allocator_warmup(
        _model: object, _device_map: object, _quantizer: object
    ) -> None:
        caching_allocator_warmup.calls += 1

    caching_allocator_warmup.__module__ = "transformers.modeling_utils"
    caching_allocator_warmup.calls = 0

    owner = types.SimpleNamespace(
        __name__="transformers.modeling_utils",
        caching_allocator_warmup=caching_allocator_warmup,
    )
    return owner, caching_allocator_warmup


class ReferenceHelperTests(unittest.TestCase):
    """Exercise deterministic evidence and publication without ML imports."""

    def test_import_installs_all_offline_and_kernel_guards(self) -> None:
        self.assertEqual(
            {
                name: REFERENCE.os.environ[name]
                for name in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                    "TOKENIZERS_PARALLELISM",
                    "HF_DEACTIVATE_ASYNC_LOAD",
                    "PYTORCH_ENABLE_MPS_FALLBACK",
                    "USE_HUB_KERNELS",
                )
            },
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_DEACTIVATE_ASYNC_LOAD": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                "USE_HUB_KERNELS": "0",
            },
        )

    def test_sha_and_f32_encoding_match_standard_vectors(self) -> None:
        self.assertEqual(
            REFERENCE.sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        self.assertEqual(
            REFERENCE.f32le_bytes([1.0, -0.0, 2.5]),
            struct.pack("<fff", 1.0, -0.0, 2.5),
        )

    def test_top_k_is_stable_and_rejects_non_finite_values(self) -> None:
        self.assertEqual(
            REFERENCE.top_k_records([1.0, 3.0, 3.0, -1.0], 3),
            [
                {"token_id": 1, "logit": 3.0},
                {"token_id": 2, "logit": 3.0},
                {"token_id": 0, "logit": 1.0},
            ],
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            REFERENCE.top_k_records([1.0, float("nan")], 1)
        with self.assertRaisesRegex(ValueError, "nonempty and finite"):
            REFERENCE.vector_summary([], )

    def test_json_integer_accepts_exact_values_only(self) -> None:
        for value, expected in ((0, 0), (-0.0, 0), (59_553_253_376.0, 59_553_253_376)):
            with self.subTest(value=value):
                self.assertEqual(
                    REFERENCE.checked_json_nonnegative_integer(value, "total_size"),
                    expected,
                )
        for value in (True, None, "1", 1.5, -1, float("inf"), 1 << 53):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "total_size"):
                    REFERENCE.checked_json_nonnegative_integer(value, "total_size")

    def test_strict_json_is_sorted_and_rejects_non_finite_numbers(self) -> None:
        self.assertEqual(
            REFERENCE.strict_json_bytes({"z": 2, "a": 1}),
            b'{\n  "a": 1,\n  "z": 2\n}\n',
        )
        with self.assertRaisesRegex(ValueError, "Out of range"):
            REFERENCE.strict_json_bytes({"bad": float("inf")})

    def test_binary_descriptor_binds_shape_length_and_digest(self) -> None:
        payload = struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
        descriptor = REFERENCE.binary_descriptor(
            "fixture.f32le", payload, [2, 2], source_dtype="float16"
        )
        self.assertEqual(descriptor["elements"], 4)
        self.assertEqual(descriptor["bytes"], 16)
        self.assertEqual(descriptor["sha256"], REFERENCE.sha256_bytes(payload))
        self.assertEqual(descriptor["source_dtype"], "float16")
        self.assertEqual(descriptor["storage_dtype"], "float32")
        self.assertTrue(descriptor["finite"])
        with self.assertRaisesRegex(ValueError, "inconsistent shape"):
            REFERENCE.binary_descriptor(
                "fixture.f32le", payload, [3], source_dtype="float16"
            )

    def test_binary16_stability_requires_a_strict_two_sided_budget(self) -> None:
        record = REFERENCE.decision_record(0, [7, 6], [8.046875, 7.96484375], [8])
        self.assertEqual(record["margin"], 0.08203125)
        self.assertGreater(record["margin_to_epsilon_ratio"], 2.0)
        with self.assertRaisesRegex(RuntimeError, "not stable"):
            REFERENCE.decision_record(1, [7, 6], [16.0, 15.984375], [8])
        with self.assertRaisesRegex(RuntimeError, "reached EOS"):
            REFERENCE.decision_record(2, [8, 6], [2.0, 1.0], [8])
        with self.assertRaisesRegex(ValueError, "exactly binary16"):
            REFERENCE.binary16_ulp(0.1)

    def test_artifact_manifest_digest_is_pinned_and_order_independent(self) -> None:
        artifacts = [
            {"relative_path": name, "bytes": size, "sha256": digest}
            for name, (size, digest) in REFERENCE.EXPECTED_ARTIFACTS.items()
        ]
        self.assertEqual(
            REFERENCE.canonical_artifact_digest(artifacts),
            REFERENCE.EXPECTED_ARTIFACT_MANIFEST_SHA256,
        )
        self.assertEqual(
            REFERENCE.canonical_artifact_digest(artifacts),
            REFERENCE.canonical_artifact_digest(list(reversed(artifacts))),
        )
        self.assertEqual(len(REFERENCE.EXPECTED_ARTIFACTS), 21)
        self.assertEqual(
            sum(
                1
                for name in REFERENCE.EXPECTED_ARTIFACTS
                if name.endswith(".safetensors")
            ),
            14,
        )

    def test_safetensors_header_parser_binds_schema_ranges_and_file_size(self) -> None:
        def encoded(header: object, tensor_payload: bytes) -> bytes:
            raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
            raw += b" " * (-len(raw) % 8)
            return struct.pack("<Q", len(raw)) + raw + tensor_payload

        header = {
            "__metadata__": {"format": "pt"},
            "base.weight": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, 4],
            },
            "mtp.buffer": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [4, 8],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"
            payload = encoded(header, b"\0" * 8)
            path.write_bytes(payload)
            tensors, record = REFERENCE.read_safetensors_header(path, len(payload))
            self.assertEqual(set(tensors), {"base.weight", "mtp.buffer"})
            self.assertEqual(record["physical_tensors"], 2)
            self.assertEqual(record["tensor_payload_bytes"], 8)
            self.assertEqual(record["metadata"], {"format": "pt"})

            bad = dict(header)
            bad["mtp.buffer"] = {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [5, 9],
            }
            bad_payload = encoded(bad, b"\0" * 9)
            path.write_bytes(bad_payload)
            with self.assertRaisesRegex(ValueError, "not gap-free"):
                REFERENCE.read_safetensors_header(path, len(bad_payload))

            bad = dict(header)
            bad["mtp.buffer"] = {
                "dtype": "F16",
                "shape": [2],
                "data_offsets": [4, 8],
            }
            bad_payload = encoded(bad, b"\0" * 8)
            path.write_bytes(bad_payload)
            with self.assertRaisesRegex(ValueError, "unsupported checkpoint dtype"):
                REFERENCE.read_safetensors_header(path, len(bad_payload))

            path.write_bytes(struct.pack("<Q", 7) + b"{" + b" " * 6)
            with self.assertRaisesRegex(ValueError, "header length"):
                REFERENCE.read_safetensors_header(path, 15)

    def test_header_inventory_digest_is_order_independent_and_range_sensitive(self) -> None:
        tensors = {
            "z": {
                "shard": "s2",
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            },
            "a": {
                "shard": "s1",
                "dtype": "BF16",
                "shape": [2, 3],
                "data_offsets": [8, 20],
            },
        }
        digest = REFERENCE.canonical_header_inventory_digest(tensors)
        self.assertEqual(
            digest,
            REFERENCE.canonical_header_inventory_digest(
                dict(reversed(list(tensors.items())))
            ),
        )
        changed = {name: dict(item) for name, item in tensors.items()}
        changed["z"]["data_offsets"] = [4, 8]
        self.assertNotEqual(
            digest, REFERENCE.canonical_header_inventory_digest(changed)
        )

    def test_pinned_base_mtp_accounting_contract_is_explicit(self) -> None:
        self.assertEqual(REFERENCE.REFERENCE_CLASS, "transformers.NemotronHForCausalLM")
        self.assertEqual(REFERENCE.EXPECTED_BASE_PHYSICAL_TENSORS, 6_243)
        self.assertEqual(REFERENCE.EXPECTED_MTP_PHYSICAL_TENSORS, 270)
        self.assertEqual(REFERENCE.EXPECTED_PHYSICAL_TENSORS, 6_513)
        self.assertEqual(
            REFERENCE.EXPECTED_BASE_LOADED_BYTES
            + REFERENCE.EXPECTED_MTP_TENSOR_BYTES,
            REFERENCE.EXPECTED_PHYSICAL_TENSOR_BYTES,
        )
        self.assertEqual(
            REFERENCE.EXPECTED_REPORTED_TENSOR_BYTES
            - REFERENCE.EXPECTED_PHYSICAL_TENSOR_BYTES,
            REFERENCE.EXPECTED_REPORTED_SIZE_DELTA,
        )

    def test_runtime_and_dependency_contract_matches_the_audited_environment(self) -> None:
        self.assertEqual(
            REFERENCE.EXPECTED_PYTHON_VERSION,
            (3, 11, 9, "final", 0),
        )
        self.assertEqual(REFERENCE.REFERENCE_SOURCE_VERSION, "5.17.0.dev0")
        self.assertEqual(
            REFERENCE.CHECKPOINT_CONFIG_TRANSFORMERS_VERSION,
            "4.57.6",
        )
        self.assertEqual(
            REFERENCE.REFERENCE_SOURCES,
            {
                "src/transformers/models/nemotron_h/modeling_nemotron_h.py": (
                    50_797,
                    "d42067429e3268d1163579f5d257fe0baa9a2b02edba5fcb175fb8d42c9b6af4",
                ),
                "src/transformers/models/nemotron_h/configuration_nemotron_h.py": (
                    13_031,
                    "00a4c60bd302d796b25c6ae25c608a5ee7e27f9a0adea92911b0c45470232052",
                ),
            },
        )
        self.assertEqual(
            REFERENCE.EXPECTED_DEPENDENCY_VERSIONS,
            {
                "accelerate": "1.12.0",
                "filelock": "3.17.0",
                "huggingface-hub": "1.5.0",
                "numpy": "2.4.3",
                "packaging": "25.0",
                "PyYAML": "6.0.2",
                "regex": "2026.1.15",
                "safetensors": "0.8.0",
                "tokenizers": "0.23.1",
                "torch": "2.10.0",
                "tqdm": "4.67.1",
            },
        )

    def test_distribution_manifest_digest_is_pinned_and_order_independent(self) -> None:
        summaries = {
            name: {
                "version": REFERENCE.EXPECTED_DEPENDENCY_VERSIONS[name],
                "verified_files": fields[0],
                "verified_bytes": fields[1],
                "ignored_bytecode_files": fields[2],
                "excluded_external_scripts": list(fields[3]),
                "payload_manifest_sha256": fields[4],
            }
            for name, fields in REFERENCE.EXPECTED_DISTRIBUTION_PAYLOADS.items()
        }
        self.assertEqual(
            REFERENCE.canonical_distribution_digest(summaries),
            REFERENCE.EXPECTED_DISTRIBUTION_MANIFEST_SHA256,
        )
        self.assertEqual(
            REFERENCE.canonical_distribution_digest(summaries),
            REFERENCE.canonical_distribution_digest(
                dict(reversed(list(summaries.items())))
            ),
        )

    def test_loading_diagnostics_require_all_empty_classes(self) -> None:
        expected_empty = REFERENCE.sha256_bytes(b"")
        summary = REFERENCE.loading_info_summary({})
        self.assertTrue(all(item == {"count": 0, "sha256": expected_empty} for item in summary.values()))
        with self.assertRaisesRegex(RuntimeError, "unexpected_keys"):
            REFERENCE.loading_info_summary({"unexpected_keys": ["model.bad"]})

    def test_hidden_trace_covers_embedding_blocks_and_final_norm(self) -> None:
        class FakeTensor:
            def __init__(self, shape: list[int]) -> None:
                self.shape = shape
                self.dtype = "float16"
                self.device = types.SimpleNamespace(type="mps")

            def __getitem__(self, _index: object) -> "FakeTensor":
                return self

            def float(self) -> "FakeTensor":
                return self

            def cpu(self) -> "FakeTensor":
                return self

            def contiguous(self) -> "FakeTensor":
                return self

        class FakeHandle:
            def __init__(self, module: "FakeModule") -> None:
                self.module = module

            def remove(self) -> None:
                self.module.hook = None

        class FakeModule:
            def __init__(self) -> None:
                self.hook: object | None = None

            def register_forward_hook(self, hook: object) -> FakeHandle:
                self.hook = hook
                return FakeHandle(self)

            def emit(self) -> None:
                assert callable(self.hook)
                self.hook(
                    self,
                    (),
                    FakeTensor([1, len(REFERENCE.INPUT_IDS), REFERENCE.HIDDEN_SIZE]),
                )

        embeddings = FakeModule()
        layers = [FakeModule() for _ in range(52)]
        final_norm = FakeModule()
        language_model = types.SimpleNamespace(
            embeddings=embeddings,
            layers=layers,
            norm_f=final_norm,
        )
        fake_torch = types.SimpleNamespace(
            float16="float16",
            stack=lambda rows: FakeTensor([len(rows), REFERENCE.HIDDEN_SIZE]),
        )
        sentinel = object()

        def run() -> object:
            for module in [embeddings, *layers, final_norm]:
                module.emit()
            return sentinel

        output, labels, trace = REFERENCE.capture_last_hidden_trace(
            fake_torch, language_model, len(REFERENCE.INPUT_IDS), run
        )
        self.assertIs(output, sentinel)
        self.assertEqual(labels[0], "embeddings")
        self.assertEqual(labels[-1], "norm_f")
        self.assertEqual(len(labels), 54)
        self.assertEqual(trace.shape, [54, REFERENCE.HIDDEN_SIZE])
        self.assertTrue(
            all(module.hook is None for module in [embeddings, *layers, final_norm])
        )

    def test_position_capture_requires_the_exact_implicit_row(self) -> None:
        class FakePositionTensor:
            shape = [1, 3]
            dtype = "int64"
            device = types.SimpleNamespace(type="mps")

            def __getitem__(self, _index: object) -> "FakePositionTensor":
                return self

            def cpu(self) -> "FakePositionTensor":
                return self

            def tolist(self) -> list[int]:
                return [4, 5, 6]

        class FakeHandle:
            def __init__(self, module: "FakeAttention") -> None:
                self.module = module

            def remove(self) -> None:
                self.module.hook = None

        class FakeAttention:
            hook: object | None = None

            def register_forward_pre_hook(
                self, hook: object, *, with_kwargs: bool
            ) -> FakeHandle:
                if with_kwargs is not True:
                    raise AssertionError("position capture must observe keyword arguments")
                self.hook = hook
                return FakeHandle(self)

            def emit(self) -> None:
                assert callable(self.hook)
                self.hook(self, (), {"position_ids": FakePositionTensor()})

        attention = FakeAttention()
        sentinel = object()

        def run() -> object:
            attention.emit()
            return sentinel

        output = REFERENCE.capture_implicit_position_ids(
            types.SimpleNamespace(int64="int64"), attention, [4, 5, 6], run
        )
        self.assertIs(output, sentinel)
        self.assertIsNone(attention.hook)
        with self.assertRaisesRegex(RuntimeError, "implicit position IDs changed"):
            REFERENCE.capture_implicit_position_ids(
                types.SimpleNamespace(int64="int64"), attention, [4, 5, 7], run
            )

    def test_committed_generation_and_cache_schedule_are_exact(self) -> None:
        # The empty continuation is a deliberate first-run fail-closed gate;
        # it must be replaced with 32 reviewed IDs before a bundle can publish.
        self.assertEqual(REFERENCE.EXPECTED_GENERATED_TOKEN_IDS, [])
        self.assertEqual(REFERENCE.EXPECTED_LAYER_TYPES.count("linear_attention"), 23)
        self.assertEqual(REFERENCE.EXPECTED_LAYER_TYPES.count("moe"), 23)
        self.assertEqual(REFERENCE.EXPECTED_LAYER_TYPES.count("full_attention"), 6)

    def test_official_cache_validator_checks_every_layer_geometry(self) -> None:
        class FakeTensor:
            def __init__(self, shape: list[int], dtype: str) -> None:
                self.shape = shape
                self.dtype = dtype
                self.device = types.SimpleNamespace(type="mps")

        class DynamicLayer:
            def __init__(self, length: int) -> None:
                self.length = length
                self.keys = FakeTensor([1, 2, length, 128], "float16")
                self.values = FakeTensor([1, 2, length, 128], "float16")

            def get_seq_length(self) -> int:
                return self.length

        class LinearAttentionLayer:
            def __init__(self, initialized: bool) -> None:
                self.number_of_states = 1
                self.record_past = False
                self.conv_states = {0: None}
                self.recurrent_states = {0: None}
                self.is_conv_states_initialized = {0: False}
                self.is_recurrent_states_initialized = {0: False}
                self.has_previous_state = {0: False}
                self.conv_kernel_size = {0: None}
                self.device = None
                self.dtype = None
                if initialized:
                    self.conv_states[0] = FakeTensor([1, 6_144, 4], "float16")
                    self.recurrent_states[0] = FakeTensor(
                        [1, 64, 64, 128], "float32"
                    )
                    self.is_conv_states_initialized[0] = True
                    self.is_recurrent_states_initialized[0] = True
                    self.has_previous_state[0] = True
                    self.conv_kernel_size[0] = 4
                    self.device = types.SimpleNamespace(type="mps")
                    self.dtype = "float16"

        class FakeDynamicCache:
            def __init__(self, length: int) -> None:
                self.length = length
                # The pinned DynamicCache reports this sentinel because the
                # concrete layers do not expose an optional batch_size field.
                self.batch_size = -1
                self.layers = [
                    DynamicLayer(length)
                    if kind == "full_attention"
                    else LinearAttentionLayer(kind == "linear_attention")
                    for kind in REFERENCE.EXPECTED_LAYER_TYPES
                ]

            def get_seq_length(self) -> int:
                return self.length

        cache = FakeDynamicCache(7)
        geometry = REFERENCE.validate_official_cache(
            types.SimpleNamespace(float16="float16", float32="float32"),
            cache,
            FakeDynamicCache,
            7,
        )
        self.assertEqual(geometry["mamba_layer_indices"][0], 0)
        self.assertEqual(geometry["attention_layer_indices"], [5, 12, 19, 26, 33, 42])
        self.assertEqual(geometry["moe_placeholder_indices"][0], 1)
        cache.layers[19].keys.shape = [2, 2, 7, 128]
        with self.assertRaisesRegex(RuntimeError, "layer 19 keys geometry changed"):
            REFERENCE.validate_official_cache(
                types.SimpleNamespace(float16="float16", float32="float32"),
                cache,
                FakeDynamicCache,
                7,
            )
        cache.layers[19].keys.shape = [1, 2, 7, 128]
        cache.layers[0].recurrent_states[0].dtype = "float16"
        with self.assertRaisesRegex(RuntimeError, "recurrent geometry changed"):
            REFERENCE.validate_official_cache(
                types.SimpleNamespace(float16="float16", float32="float32"),
                cache,
                FakeDynamicCache,
                7,
            )
        cache.layers[0].recurrent_states[0].dtype = "float32"
        cache.layers[1].conv_states[0] = FakeTensor([1, 1, 1], "float16")
        with self.assertRaisesRegex(RuntimeError, "placeholder 1 was initialized"):
            REFERENCE.validate_official_cache(
                types.SimpleNamespace(float16="float16", float32="float32"),
                cache,
                FakeDynamicCache,
                7,
            )

    def test_allocator_override_is_scoped_exactly_once_to_mps(self) -> None:
        owner, original = fake_modeling_utils()
        sentinel = object()

        def loader() -> object:
            owner.caching_allocator_warmup(
                FakeNemotronModel(),
                {
                    f"parameter.{index}": "mps"
                    for index in range(REFERENCE.EXPECTED_ALLOCATOR_WARMUP_ENTRIES)
                },
                None,
            )
            return sentinel

        loaded, provenance = REFERENCE.load_with_verified_mps_warmup(
            owner, FakeTorch, FakeNemotronModel, loader
        )
        self.assertIs(loaded, sentinel)
        self.assertIs(owner.caching_allocator_warmup, original)
        self.assertEqual(original.calls, 1)
        self.assertEqual(provenance["call_count"], 1)
        self.assertEqual(
            provenance["expanded_device_map_entries"],
            REFERENCE.EXPECTED_ALLOCATOR_WARMUP_ENTRIES,
        )
        self.assertTrue(provenance["restored"])

        def mixed_loader() -> None:
            owner.caching_allocator_warmup(
                FakeNemotronModel(),
                {
                    **{
                        f"parameter.{index}": "mps"
                        for index in range(
                            REFERENCE.EXPECTED_ALLOCATOR_WARMUP_ENTRIES - 1
                        )
                    },
                    "escaped": "cpu",
                },
                None,
            )

        with self.assertRaisesRegex(RuntimeError, "escaped MPS"):
            REFERENCE.load_with_verified_mps_warmup(
                owner, FakeTorch, FakeNemotronModel, mixed_loader
            )
        self.assertIs(owner.caching_allocator_warmup, original)

    def test_write_create_only_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence"
            REFERENCE.write_create_only(path, b"first")
            with self.assertRaises(FileExistsError):
                REFERENCE.write_create_only(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_bundle_publication_is_atomic_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                REFERENCE.JSON_FILENAME: b"json",
                REFERENCE.LOGITS_FILENAME: b"logits",
                REFERENCE.HIDDEN_FILENAME: b"hidden",
                REFERENCE.CACHED_LOGITS_FILENAME: b"cached-logits",
                REFERENCE.TEACHER_LOGITS_FILENAME: b"teacher-logits",
                REFERENCE.CACHED_PREFILL_HIDDEN_FILENAME: b"prefill-hidden",
                REFERENCE.CACHED_DECODE_HIDDEN_FILENAME: b"decode-hidden",
            }
            bundle = REFERENCE.publish_reference_bundle(root, payloads)
            self.assertEqual(bundle, root / REFERENCE.BUNDLE_DIRNAME)
            self.assertEqual(
                {path.name: path.read_bytes() for path in bundle.iterdir()}, payloads
            )
            with self.assertRaises(FileExistsError):
                REFERENCE.publish_reference_bundle(root, payloads)
            self.assertEqual(list(root.glob(".*.staging-*")), [])

    def test_failed_publication_keeps_forensic_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                REFERENCE.JSON_FILENAME: b"json",
                REFERENCE.LOGITS_FILENAME: b"logits",
                REFERENCE.HIDDEN_FILENAME: b"hidden",
                REFERENCE.CACHED_LOGITS_FILENAME: b"cached-logits",
                REFERENCE.TEACHER_LOGITS_FILENAME: b"teacher-logits",
                REFERENCE.CACHED_PREFILL_HIDDEN_FILENAME: b"prefill-hidden",
                REFERENCE.CACHED_DECODE_HIDDEN_FILENAME: b"decode-hidden",
            }
            original = REFERENCE.write_create_only

            def fail_hidden(path: Path, payload: bytes) -> None:
                if path.name == REFERENCE.HIDDEN_FILENAME:
                    raise OSError("injected failure")
                original(path, payload)

            with mock.patch.object(REFERENCE, "write_create_only", side_effect=fail_hidden):
                with self.assertRaisesRegex(OSError, "injected failure") as raised:
                    REFERENCE.publish_reference_bundle(root, payloads)
            self.assertFalse((root / REFERENCE.BUNDLE_DIRNAME).exists())
            staging = list(root.glob(".*.staging-*"))
            self.assertEqual(len(staging), 1)
            self.assertTrue(
                any("recoverable reference staging" in note for note in raised.exception.__notes__)
            )

    def test_external_paths_are_disjoint_and_bundle_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, source, output = root / "model", root / "source", root / "output"
            for path in (model, source, output):
                path.mkdir()
            observed = REFERENCE.ensure_external_paths(model, source, output)
            self.assertEqual(observed[-1], output.resolve() / REFERENCE.BUNDLE_DIRNAME)
            observed[-1].mkdir()
            with self.assertRaises(FileExistsError):
                REFERENCE.ensure_external_paths(model, source, output)

    def test_checkpoint_root_inventory_is_closed_over_loader_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.dict(
                REFERENCE.EXPECTED_ARTIFACTS,
                {"config.json": (2, "0" * 64)},
                clear=True,
            ), mock.patch.dict(
                REFERENCE.EXPECTED_INERT_ROOT_FILES,
                {"README.md": 5},
                clear=True,
            ):
                (root / "config.json").write_bytes(b"{}")
                (root / "README.md").write_text("inert", encoding="utf-8")
                (root / ".cache").mkdir()
                REFERENCE.verify_checkpoint_root_inventory(root)

                (root / "model.safetensors").write_bytes(b"injected")
                with self.assertRaisesRegex(RuntimeError, "could influence local loading"):
                    REFERENCE.verify_checkpoint_root_inventory(root)

    def test_cli_is_mps_only_and_has_no_device_or_network_switch(self) -> None:
        parsed = REFERENCE.parse_arguments(
            ["--model-dir", "m", "--transformers-dir", "t", "--output-dir", "o"]
        )
        self.assertFalse(hasattr(parsed, "device"))
        with mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                REFERENCE.parse_arguments(
                    [
                        "--model-dir",
                        "m",
                        "--transformers-dir",
                        "t",
                        "--output-dir",
                        "o",
                        "--device",
                        "cuda",
                    ]
                )

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                REFERENCE.json_without_duplicate_keys(path)

    def test_checked_duration_rejects_negative_and_non_finite_values(self) -> None:
        self.assertEqual(REFERENCE.checked_duration("fixture", 0.0), 0.0)
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "duration is invalid"):
                    REFERENCE.checked_duration("fixture", value)


if __name__ == "__main__":
    unittest.main()
