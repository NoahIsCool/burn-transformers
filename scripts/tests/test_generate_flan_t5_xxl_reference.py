"""Pure helper tests for the opt-in FLAN-T5-XXL reference generator."""

from __future__ import annotations

import importlib.util
import base64
import io
import json
import math
import struct
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "generate_flan_t5_xxl_reference.py"
SPEC = importlib.util.spec_from_file_location("generate_flan_t5_xxl_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class FakeTorch:
    """Minimal ``torch.device`` facade for allocator-override tests."""

    @staticmethod
    def device(assignment: object) -> object:
        """Return only the type prefix of a string device assignment."""

        if not isinstance(assignment, str) or not assignment:
            raise TypeError("invalid fake device")
        return types.SimpleNamespace(type=assignment.partition(":")[0])


class FakeT5Model:
    """Exact model sentinel accepted by the allocator wrapper."""


def fake_modeling_utils() -> tuple[object, object]:
    """Return an upstream-shaped owner and its original warmup function."""

    def caching_allocator_warmup(
        _model: object, _expanded_device_map: object, _hf_quantizer: object
    ) -> None:
        """Fail if the wrapper does not intercept this original function."""

        raise AssertionError("original warmup must be intercepted")

    caching_allocator_warmup.__module__ = "transformers.modeling_utils"
    owner = types.SimpleNamespace(
        __name__="transformers.modeling_utils",
        caching_allocator_warmup=caching_allocator_warmup,
    )
    return owner, caching_allocator_warmup


def synthetic_safetensors_bytes(
    header: object, tensor_payload: bytes = b""
) -> bytes:
    """Serialize a small header envelope understood by the pure parser."""

    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + tensor_payload


class ReferenceHelperTests(unittest.TestCase):
    """Exercise safety and canonicalization without loading Torch or XXL."""

    def test_sha256_helpers_match_standard_vectors(self) -> None:
        """Require both memory and ranged-file hashing to use exact bytes."""

        self.assertEqual(
            REFERENCE.sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload"
            path.write_bytes(b"0123456789")
            self.assertEqual(
                REFERENCE.sha256_file_range(path, 2, 4),
                REFERENCE.sha256_bytes(b"2345"),
            )
            with self.assertRaisesRegex(EOFError, "short read"):
                REFERENCE.sha256_file_range(path, 8, 4)
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                REFERENCE.sha256_file_range(path, -1, 1)

    def test_json_parser_rejects_duplicate_keys(self) -> None:
        """Ensure control files and SafeTensors headers cannot be ambiguous."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                REFERENCE.json_without_duplicate_keys(path)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            REFERENCE.json_bytes_without_duplicate_keys(b'{"a":1,"a":2}', "header")

    def test_checked_integer_accepts_only_exact_interoperable_values(self) -> None:
        """Canonicalize integral floats but reject booleans and ambiguity."""

        for encoded, expected in (
            (0, 0),
            (-0.0, 0),
            (123.0, 123),
            (REFERENCE.MAX_EXACT_JSON_INTEGER, REFERENCE.MAX_EXACT_JSON_INTEGER),
        ):
            with self.subTest(encoded=encoded):
                observed = REFERENCE.checked_nonnegative_integer(encoded, "fixture")
                self.assertIs(type(observed), int)
                self.assertEqual(observed, expected)
        for invalid in (
            True,
            None,
            "1",
            -1,
            1.5,
            float("nan"),
            float("inf"),
            REFERENCE.MAX_EXACT_JSON_INTEGER + 1,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "fixture"):
                    REFERENCE.checked_nonnegative_integer(invalid, "fixture")

    def test_checked_product_rejects_bad_dimensions_and_overflow(self) -> None:
        """Protect tensor byte arithmetic from invalid shapes and overflow."""

        self.assertEqual(REFERENCE.checked_product([2, 3, 4], "shape"), 24)
        self.assertEqual(REFERENCE.checked_product([], "shape"), 1)
        for invalid in ([2, -1], [2, True], [REFERENCE.MAX_EXACT_JSON_INTEGER, 2]):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "shape"):
                    REFERENCE.checked_product(invalid, "shape")

    def test_ranking_is_finite_and_breaks_ties_by_token_id(self) -> None:
        """Make vocabulary ranking deterministic even when scores are equal."""

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
        with self.assertRaisesRegex(ValueError, "outside"):
            REFERENCE.top_k_records([1.0], 0)

    def test_vector_summary_requires_finite_unique_probes(self) -> None:
        """Reject summaries which could hide corrupt values or duplicate evidence."""

        summary = REFERENCE.vector_summary([1.0, -2.0, 3.0], [0, 2])
        self.assertTrue(summary["finite"])
        self.assertEqual(summary["sum"], 2.0)
        self.assertEqual(summary["l2_norm"], math.sqrt(14.0))
        with self.assertRaisesRegex(ValueError, "unique"):
            REFERENCE.vector_summary([1.0, 2.0], [0, 0])
        with self.assertRaisesRegex(ValueError, "finite"):
            REFERENCE.vector_summary([1.0, float("inf")], [0])

    def test_tensor_payload_is_finite_shaped_little_endian_f32(self) -> None:
        """Pin the raw matrix wire format consumed by Rust."""

        import numpy
        import torch

        payload, values = REFERENCE.tensor_payload(
            torch, numpy, torch.tensor([[1.0, -2.5]], dtype=torch.float64), [1, 2]
        )
        self.assertEqual(payload, struct.pack("<ff", 1.0, -2.5))
        self.assertEqual(values, [1.0, -2.5])
        with self.assertRaisesRegex(RuntimeError, "shape changed"):
            REFERENCE.tensor_payload(torch, numpy, torch.ones(2), [1, 2])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            REFERENCE.tensor_payload(
                torch, numpy, torch.tensor([float("nan")]), [1]
            )

    def test_tensor_parity_is_elementwise_not_only_cosine(self) -> None:
        """Reject localized corruption even when a high-dimensional cosine is high."""

        import torch

        expected = torch.ones(10_000, dtype=torch.float32)
        close = expected.clone()
        close[0] += 1.0e-5
        summary = REFERENCE.tensor_parity_summary(torch, close, expected)
        self.assertEqual(summary["failure_count"], 0)
        corrupted = expected.clone()
        corrupted[0] += 0.1
        self.assertGreater(
            REFERENCE.tensor_cosine(torch, corrupted, expected), 0.999_9
        )
        with self.assertRaisesRegex(RuntimeError, "elementwise parity failed"):
            REFERENCE.tensor_parity_summary(torch, corrupted, expected)

    def test_f32_stability_uses_both_candidates_error_bounds(self) -> None:
        """Prove top-1 invariance with the stated absolute-relative tolerance."""

        stable = REFERENCE.greedy_decision_evidence(3, [7, 6], [2.0, 1.0])
        self.assertTrue(stable["stable_under_f32_parity_tolerance"])
        self.assertAlmostEqual(stable["combined_error_bound"], 0.0005)
        self.assertEqual(
            REFERENCE.validate_decision_evidence([stable]),
            (1.0, 2000.0),
        )
        unstable = REFERENCE.greedy_decision_evidence(
            4, [7, 6], [1.0001, 1.0]
        )
        self.assertFalse(unstable["stable_under_f32_parity_tolerance"])
        with self.assertRaisesRegex(RuntimeError, r"steps \[4\]"):
            REFERENCE.validate_decision_evidence([unstable])

    def test_strict_json_rejects_non_finite_values(self) -> None:
        """Prevent implementation-specific NaN tokens in provenance output."""

        self.assertEqual(REFERENCE.strict_json_bytes({"x": 1.0}), b'{\n  "x": 1.0\n}\n')
        with self.assertRaisesRegex(ValueError, "Out of range"):
            REFERENCE.strict_json_bytes({"x": float("nan")})

    def test_artifact_and_source_digests_are_order_independent(self) -> None:
        """Make manifest identities independent of dictionary insertion order."""

        first = {"relative_path": "a", "bytes": 1, "sha256": "0" * 64}
        second = {"relative_path": "b", "bytes": 2, "sha256": "f" * 64}
        self.assertEqual(
            REFERENCE.canonical_artifact_digest([first, second]),
            REFERENCE.canonical_artifact_digest([second, first]),
        )
        source_a = {"module": "a", **first}
        source_b = {"module": "b", **second}
        self.assertEqual(
            REFERENCE.canonical_source_digest([source_a, source_b]),
            REFERENCE.canonical_source_digest([source_b, source_a]),
        )

    def test_pinned_artifact_manifest_constant_matches_literal_inventory(self) -> None:
        """Catch transcription errors without rereading the 45 GB checkpoint."""

        artifacts = [
            {"relative_path": name, "bytes": size, "sha256": digest}
            for name, (size, digest) in (
                REFERENCE.EXPECTED_CONTROL_FILES | REFERENCE.EXPECTED_SHARDS
            ).items()
        ]
        self.assertEqual(
            REFERENCE.canonical_artifact_digest(artifacts),
            REFERENCE.EXPECTED_ARTIFACT_MANIFEST_SHA256,
        )

    def test_safetensors_header_parser_bounds_the_envelope(self) -> None:
        """Parse a valid header and reject truncated or oversized declarations."""

        header = {
            "x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "__metadata__": {"format": "pt"},
        }
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.safetensors"
            valid.write_bytes(synthetic_safetensors_bytes(header, b"\0" * 8))
            observed, data_start, file_bytes = REFERENCE.read_safetensors_header(valid)
            self.assertEqual(observed, header)
            self.assertEqual(data_start + 8, file_bytes)

            truncated = Path(directory) / "truncated.safetensors"
            truncated.write_bytes(struct.pack("<Q", 100) + b"{}")
            with self.assertRaisesRegex(ValueError, "exceeds file size"):
                REFERENCE.read_safetensors_header(truncated)

            oversized = Path(directory) / "oversized.safetensors"
            oversized.write_bytes(
                struct.pack("<Q", REFERENCE.MAX_SAFETENSORS_HEADER_BYTES + 1)
            )
            with self.assertRaisesRegex(ValueError, "header length is invalid"):
                REFERENCE.read_safetensors_header(oversized)

    def test_header_inventory_digest_sorts_tensor_names(self) -> None:
        """Pin the Rust-compatible field separators and canonical sort."""

        first = {
            "shard": "s", "dtype": "F32", "shape": [2], "data_offsets": [0, 8]
        }
        second = {
            "shard": "s", "dtype": "F32", "shape": [1], "data_offsets": [8, 12]
        }
        forward = {"a": first, "b": second}
        reverse = {"b": second, "a": first}
        self.assertEqual(
            REFERENCE.canonical_header_inventory_digest(forward),
            REFERENCE.canonical_header_inventory_digest(reverse),
        )

    def test_regular_file_identity_rejects_paths_and_symlinks(self) -> None:
        """Prevent traversal and symlink substitution during checkpoint reads."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "target"
            target.write_bytes(b"x")
            self.assertEqual(REFERENCE.regular_file_identity(root, "target")[1], 1)
            for invalid in ("../target", "/target", "./target"):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "local basename"):
                        REFERENCE.regular_file_identity(root, invalid)
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "nonsymlink"):
                REFERENCE.regular_file_identity(root, "link")

    def test_external_paths_reject_nested_outputs_and_dangling_bundle_links(self) -> None:
        """Resolve roots strictly and treat dangling final symlinks as occupied."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            output = root / "output"
            nested = model / "nested"
            model.mkdir()
            output.mkdir()
            nested.mkdir()
            observed_model, observed_output, bundle = REFERENCE.ensure_external_paths(
                model, output
            )
            self.assertEqual(observed_model, model.resolve())
            self.assertEqual(observed_output, output.resolve())
            with self.assertRaisesRegex(ValueError, "outside the checkpoint"):
                REFERENCE.ensure_external_paths(model, nested)
            bundle.symlink_to(root / "missing")
            with self.assertRaises(FileExistsError):
                REFERENCE.ensure_external_paths(model, output)

    def test_closed_root_inventory_rejects_loading_overlays(self) -> None:
        """Accept only pinned consumed/inert names and reject loader precedence attacks."""

        controls = {"config.json": (1, "unused")}
        shards = {"model-00001.safetensors": (2, "unused")}
        ignored = {"README.md": 3}
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(REFERENCE, "EXPECTED_CONTROL_FILES", controls),
            mock.patch.object(REFERENCE, "EXPECTED_SHARDS", shards),
            mock.patch.object(REFERENCE, "EXPECTED_IGNORED_ROOT_FILES", ignored),
            mock.patch.object(
                REFERENCE, "ALLOWED_IGNORED_ROOT_DIRECTORIES", {".cache"}
            ),
        ):
            root = Path(directory)
            (root / "config.json").write_bytes(b"x")
            (root / "model-00001.safetensors").write_bytes(b"xx")
            (root / "README.md").write_bytes(b"xxx")
            (root / ".cache").mkdir()
            inventory = REFERENCE.verify_root_inventory(root)
            self.assertEqual(len(inventory), 4)

            overlay = root / "model.safetensors"
            overlay.write_bytes(b"injected")
            with self.assertRaisesRegex(ValueError, "could influence"):
                REFERENCE.verify_root_inventory(root)
            overlay.unlink()

            (root / "README.md").unlink()
            (root / "README.md").symlink_to(root / "config.json")
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                REFERENCE.verify_root_inventory(root)

    def test_long_form_fixture_is_distinct_and_exactly_pinned(self) -> None:
        """Keep the 32-token gate independent from the short beam fixture."""

        self.assertNotEqual(REFERENCE.LONG_FORM_PROMPT, REFERENCE.TRANSLATION_PROMPT)
        self.assertNotEqual(
            REFERENCE.LONG_FORM_INPUT_IDS, REFERENCE.TRANSLATION_INPUT_IDS
        )
        self.assertEqual(len(REFERENCE.LONG_FORM_INPUT_IDS), 51)
        self.assertEqual(REFERENCE.LONG_FORM_INPUT_IDS[-1], REFERENCE.EOS_TOKEN_ID)
        canonical_ids = json.dumps(
            REFERENCE.LONG_FORM_INPUT_IDS, separators=(",", ":")
        ).encode("ascii")
        self.assertEqual(
            REFERENCE.sha256_bytes(canonical_ids),
            "59cf3fd60cd596578e77a7e4b57ce5209c21c1394d3bf8b8972c7988e220d194",
        )
        self.assertEqual(REFERENCE.MAX_GREEDY_TOKENS, 32)

    def test_atomic_bundle_is_create_only_and_complete(self) -> None:
        """Publish all six outputs together and refuse replacement."""

        payloads = {
            REFERENCE.ENCODER_RAW_FILENAME: b"encoder",
            REFERENCE.DECODER_RAW_FILENAME: b"decoder",
            REFERENCE.LOGITS_RAW_FILENAME: b"logits",
            REFERENCE.CACHED_LOGITS_RAW_FILENAME: b"cached",
            REFERENCE.TEACHER_LOGITS_RAW_FILENAME: b"teacher",
            REFERENCE.JSON_FILENAME: b"json",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = REFERENCE.publish_reference_bundle(root, payloads)
            self.assertEqual(bundle.name, REFERENCE.BUNDLE_DIRNAME)
            self.assertEqual(
                {path.name: path.read_bytes() for path in bundle.iterdir()}, payloads
            )
            with self.assertRaises(FileExistsError):
                REFERENCE.publish_reference_bundle(root, payloads)
            incomplete = dict(payloads)
            incomplete.pop(REFERENCE.JSON_FILENAME)
            with self.assertRaisesRegex(ValueError, "payload names differ"):
                REFERENCE.publish_reference_bundle(Path(directory) / "unused", incomplete)

    def test_allocator_override_is_exact_scoped_and_recorded(self) -> None:
        """Intercept one all-MPS unquantized call and restore the original."""

        owner, original = fake_modeling_utils()
        sentinel = object()

        def loader() -> object:
            """Invoke the patched hook exactly as the upstream loader does."""

            owner.caching_allocator_warmup(
                FakeT5Model(), {"a": "mps", "b": "mps:0"}, None
            )
            return sentinel

        loaded, provenance = REFERENCE.load_with_verified_mps_allocator_warmup_noop(
            owner, FakeTorch, FakeT5Model, loader
        )
        self.assertIs(loaded, sentinel)
        self.assertIs(owner.caching_allocator_warmup, original)
        self.assertEqual(provenance["call_count"], 1)
        self.assertEqual(provenance["expanded_device_map_entries"], 2)
        self.assertTrue(provenance["restored"])

    def test_allocator_override_restores_after_loader_failure(self) -> None:
        """Never leave a process-wide monkey patch installed after an exception."""

        owner, original = fake_modeling_utils()

        def loader() -> object:
            """Exercise the hook before simulating a loader failure."""

            owner.caching_allocator_warmup(FakeT5Model(), {"a": "mps"}, None)
            raise RuntimeError("load failed")

        with self.assertRaisesRegex(RuntimeError, "load failed"):
            REFERENCE.load_with_verified_mps_allocator_warmup_noop(
                owner, FakeTorch, FakeT5Model, loader
            )
        self.assertIs(owner.caching_allocator_warmup, original)

    def test_generation_config_records_defaults_and_explicit_overrides(self) -> None:
        """Serialize the complete effective beam policy, including implicit defaults."""

        from transformers import GenerationConfig

        base = GenerationConfig(
            decoder_start_token_id=0, eos_token_id=1, pad_token_id=0
        )
        effective, record = REFERENCE.normalized_generation_config(
            base,
            {
                "num_beams": 4,
                "max_new_tokens": 16,
                "do_sample": False,
                "early_stopping": True,
            },
        )
        self.assertEqual(effective.num_beams, 4)
        self.assertEqual(record["num_beams"], 4)
        self.assertEqual(record["length_penalty"], 1.0)
        self.assertEqual(record["decoder_start_token_id"], 0)
        self.assertIn("num_return_sequences", record)

    def test_loading_diagnostics_reject_any_unexplained_key(self) -> None:
        """Keep official loader classifications strict and canonically hashed."""

        empty = REFERENCE.loading_info_summary({})
        self.assertTrue(all(value["count"] == 0 for value in empty.values()))
        with self.assertRaisesRegex(RuntimeError, "missing_keys"):
            REFERENCE.loading_info_summary({"missing_keys": ["x.weight"]})

    def test_memory_tracker_records_mps_samples_and_process_peak(self) -> None:
        """Expose Metal's sampled peak honestly alongside the kernel RSS peak."""

        class FakeMps:
            """Mutable allocator counters for deterministic sampling."""

            current = 10
            driver = 20

            @classmethod
            def current_allocated_memory(cls) -> int:
                """Return the current fake allocation."""

                return cls.current

            @classmethod
            def driver_allocated_memory(cls) -> int:
                """Return the current fake driver allocation."""

                return cls.driver

        fake_torch = types.SimpleNamespace(mps=FakeMps)
        tracker = REFERENCE.RuntimeMemoryTracker(fake_torch, interval_seconds=0.5)
        tracker.start()
        FakeMps.current = 30
        FakeMps.driver = 40
        tracker.sample()
        result = tracker.stop()
        self.assertEqual(result["mps_current_allocated_sampled_peak_bytes"], 30)
        self.assertEqual(result["mps_driver_allocated_sampled_peak_bytes"], 40)
        self.assertGreater(result["process_peak_rss_bytes"], 0)
        self.assertGreaterEqual(result["sample_count"], 3)

    def test_mps_module_and_cache_inventory_reject_device_escape(self) -> None:
        """Validate every recurrent tensor rather than trusting the device map."""

        float32 = object()

        class FakeTruth:
            """Torch-like finite reduction result."""

            def all(self) -> "FakeTruth":
                """Return the scalar reduction facade."""

                return self

            @staticmethod
            def item() -> bool:
                """Report a finite tensor."""

                return True

        class FakeTensor:
            """Minimal tensor device/shape facade."""

            def __init__(self, shape: list[int], device: str = "mps") -> None:
                self.shape = shape
                self.device = types.SimpleNamespace(type=device)
                self.dtype = float32

            def numel(self) -> int:
                """Return the exact shape product."""

                return math.prod(self.shape)

        class DeviceTorch:
            """Torch facade used by MPS inventory helpers."""

            pass

        DeviceTorch.float32 = float32
        DeviceTorch.is_tensor = staticmethod(lambda value: isinstance(value, FakeTensor))
        DeviceTorch.isfinite = staticmethod(lambda _value: FakeTruth())

        parameter = FakeTensor([2, 3])
        buffer = FakeTensor([4])
        model = types.SimpleNamespace(
            named_parameters=lambda: [("weight", parameter)],
            named_buffers=lambda: [("buffer", buffer)],
        )
        inventory = REFERENCE.module_device_inventory(DeviceTorch, model)
        self.assertEqual(inventory["parameter_elements"], 6)
        self.assertEqual(inventory["buffer_elements"], 4)

        self_tensor = FakeTensor([1, 64, 3, 64])
        cross_tensor = FakeTensor([1, 64, 5, 64])
        layer = (
            self_tensor,
            self_tensor,
            None,
            cross_tensor,
            cross_tensor,
            None,
        )
        cache = REFERENCE.cache_device_inventory(
            DeviceTorch,
            [layer, layer],
            expected_layers=2,
            expected_self_length=3,
            expected_cross_length=5,
        )
        self.assertEqual(cache["tensor_count"], 8)
        parameter.device.type = "cpu"
        with self.assertRaisesRegex(RuntimeError, "not on MPS"):
            REFERENCE.module_device_inventory(DeviceTorch, model)

    def test_authenticated_import_prefix_is_private_empty_and_restored(self) -> None:
        """Resolve a package first, suppress adjacent pyc, then restore interpreter state."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            root.mkdir()
            origin = root / "__init__.py"
            origin.write_text("", encoding="utf-8")
            fake_spec = types.SimpleNamespace(origin=str(origin))
            previous = (sys.dont_write_bytecode, sys.pycache_prefix)
            with (
                mock.patch.object(
                    REFERENCE, "EXPECTED_PACKAGE_VERSIONS", {"fixture": "1"}
                ),
                mock.patch.object(
                    REFERENCE.importlib.util, "find_spec", return_value=fake_spec
                ),
            ):
                cache, resolutions, saved = REFERENCE.prepare_authenticated_imports(
                    Path(directory), {"fixture": root.resolve()}
                )
                self.assertEqual(resolutions, {"fixture": str(origin.resolve())})
                bindings = REFERENCE.bind_imported_modules(
                    {"fixture": types.SimpleNamespace(__file__=str(origin))},
                    {"fixture": root.resolve()},
                )
                self.assertEqual(bindings, {"fixture": str(origin.resolve())})
                self.assertTrue(sys.dont_write_bytecode)
                self.assertEqual(sys.pycache_prefix, str(cache))
                policy = REFERENCE.finalize_authenticated_imports(cache, saved)
            self.assertEqual(
                (sys.dont_write_bytecode, sys.pycache_prefix), previous
            )
            self.assertTrue(policy["restored"])

            with (
                mock.patch.object(
                    REFERENCE, "EXPECTED_PACKAGE_VERSIONS", {"fixture": "1"}
                ),
                mock.patch.object(
                    REFERENCE.importlib.util, "find_spec", return_value=fake_spec
                ),
            ):
                cache, _, saved = REFERENCE.prepare_authenticated_imports(
                    Path(directory), {"fixture": root.resolve()}
                )
                (cache / "unexpected.pyc").write_bytes(b"x")
                with self.assertRaisesRegex(RuntimeError, "unexpectedly changed"):
                    REFERENCE.finalize_authenticated_imports(cache, saved)
            self.assertEqual(
                (sys.dont_write_bytecode, sys.pycache_prefix), previous
            )

    def test_package_and_reference_sources_match_prepared_environment(self) -> None:
        """Bind helper execution to the exact installed Transformers wheel."""

        versions, provenance, distribution_digest, import_roots = REFERENCE.verify_packages()
        self.assertEqual(versions, REFERENCE.EXPECTED_PACKAGE_VERSIONS)
        self.assertEqual(
            distribution_digest, REFERENCE.EXPECTED_DISTRIBUTION_MANIFEST_SHA256
        )
        self.assertEqual(set(provenance), set(REFERENCE.EXPECTED_PACKAGE_VERSIONS))
        self.assertEqual(set(import_roots), set(REFERENCE.EXPECTED_PACKAGE_VERSIONS))
        for name, payload in REFERENCE.EXPECTED_DISTRIBUTION_PAYLOADS.items():
            self.assertEqual(
                (
                    provenance[name]["payload"]["verified_files"],
                    provenance[name]["payload"]["verified_bytes"],
                    provenance[name]["payload"]["ignored_bytecode_files"],
                    provenance[name]["payload"]["manifest_sha256"],
                ),
                payload,
            )

        import transformers

        sources, source_digest = REFERENCE.verify_reference_sources(
            Path(transformers.__file__).resolve(strict=True).parent
        )
        self.assertEqual(
            source_digest, REFERENCE.EXPECTED_REFERENCE_SOURCE_MANIFEST_SHA256
        )
        self.assertEqual(
            {source["module"] for source in sources},
            set(REFERENCE.EXPECTED_REFERENCE_SOURCES),
        )

    def test_distribution_payload_verification_detects_record_tampering(self) -> None:
        """Authenticate actual wheel bytes, not only the RECORD manifest itself."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload_path = root / "module.py"
            payload_path.write_bytes(b"abc")
            encoded = base64.urlsafe_b64encode(
                bytes.fromhex(REFERENCE.sha256_bytes(b"abc"))
            ).decode("ascii").rstrip("=")
            payload_entry = types.SimpleNamespace(
                name="module.py",
                parent=types.SimpleNamespace(name="package"),
                size=3,
                hash=types.SimpleNamespace(mode="sha256", value=encoded),
            )
            record_entry = types.SimpleNamespace(
                name="RECORD",
                parent=types.SimpleNamespace(name="package.dist-info"),
                size=None,
                hash=None,
            )

            class FakeDistribution:
                """Map two synthetic RECORD entries to controlled local files."""

                files = [payload_entry, record_entry]

                @staticmethod
                def locate_file(entry: object) -> Path:
                    """Resolve only the authenticated payload entry."""

                    if entry is payload_entry:
                        return payload_path
                    return root / "RECORD"

            summary = REFERENCE.verified_distribution_payload(
                FakeDistribution(), "package"
            )
            self.assertEqual(summary["verified_files"], 1)
            payload_path.write_bytes(b"abd")
            with self.assertRaisesRegex(RuntimeError, "payload hash changed"):
                REFERENCE.verified_distribution_payload(FakeDistribution(), "package")

    def test_cli_help_exits_without_checkpoint_or_ml_imports(self) -> None:
        """Keep help usable without allocating or validating a large artifact."""

        stdout = io.StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit) as exit_status:
            REFERENCE.parse_arguments(["--help"])
        self.assertEqual(exit_status.exception.code, 0)
        self.assertIn("--preflight-only", stdout.getvalue())

    def test_main_preflight_returns_before_heavy_imports(self) -> None:
        """Prove preflight cannot accidentally enter the Torch execution section."""

        arguments = types.SimpleNamespace(
            model_dir=Path("model"), output_dir=Path("output"), preflight_only=True
        )
        checkpoint = {
            "physical_tensor_count": REFERENCE.EXPECTED_PHYSICAL_TENSORS,
            "checkpoint_tensor_bytes": REFERENCE.EXPECTED_CHECKPOINT_TENSOR_BYTES,
            "artifact_manifest_sha256": REFERENCE.EXPECTED_ARTIFACT_MANIFEST_SHA256,
        }
        stdout = io.StringIO()
        with (
            mock.patch.object(REFERENCE, "parse_arguments", return_value=arguments),
            mock.patch.object(
                REFERENCE,
                "ensure_external_paths",
                return_value=(Path("model"), Path("output"), Path("bundle")),
            ),
            mock.patch.object(
                REFERENCE,
                "verify_packages",
                return_value=(
                    {},
                    {},
                    REFERENCE.EXPECTED_DISTRIBUTION_MANIFEST_SHA256,
                    {"transformers": Path("transformers")},
                ),
            ),
            mock.patch.object(REFERENCE, "verify_checkpoint", return_value=checkpoint),
            mock.patch.object(
                REFERENCE,
                "verify_reference_sources",
                return_value=([], REFERENCE.EXPECTED_REFERENCE_SOURCE_MANIFEST_SHA256),
            ),
            redirect_stdout(stdout),
        ):
            self.assertEqual(REFERENCE.main([]), 0)
        self.assertIn("physical_tensors=559", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
