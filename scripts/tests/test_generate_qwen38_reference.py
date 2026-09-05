"""Pure helper tests for the large, opt-in Qwen3.8 reference generator."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import struct
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "generate_qwen38_reference.py"
SPEC = importlib.util.spec_from_file_location("generate_qwen38_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class FakeTorch:
    """Minimal torch.device facade used to test allocator policy validation."""

    @staticmethod
    def device(assignment: object) -> object:
        """Return a device-like object whose type follows a string assignment."""

        if not isinstance(assignment, str) or not assignment:
            raise TypeError("invalid fake device")
        return types.SimpleNamespace(type=assignment.partition(":")[0])


class FakeQwenModel:
    """Exact model-class sentinel accepted by allocator override helper tests."""


def fake_modeling_utils() -> tuple[object, object]:
    """Create an upstream-shaped module and retain its original warmup callable."""

    def caching_allocator_warmup(
        _model: object, _expanded_device_map: object, _hf_quantizer: object
    ) -> None:
        raise AssertionError("the original allocator warmup must be intercepted")

    caching_allocator_warmup.__module__ = "transformers.modeling_utils"
    owner = types.SimpleNamespace(
        __name__="transformers.modeling_utils",
        caching_allocator_warmup=caching_allocator_warmup,
    )
    return owner, caching_allocator_warmup


class ReferenceHelperTests(unittest.TestCase):
    """Exercise deterministic serialization and create-only safety without ML imports."""

    def test_sha256_bytes_matches_standard_vector(self) -> None:
        self.assertEqual(
            REFERENCE.sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_f32_encoding_is_little_endian(self) -> None:
        self.assertEqual(
            REFERENCE.f32le_bytes([1.0, -0.0, 2.5]),
            struct.pack("<fff", 1.0, -0.0, 2.5),
        )

    def test_top_k_breaks_ties_by_token_id(self) -> None:
        self.assertEqual(
            REFERENCE.top_k_records([1.0, 3.0, 3.0, -1.0], 3),
            [
                {"token_id": 1, "logit": 3.0},
                {"token_id": 2, "logit": 3.0},
                {"token_id": 0, "logit": 1.0},
            ],
        )

    def test_top_k_rejects_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            REFERENCE.top_k_records([1.0, float("inf")], 1)

    def test_vector_summary_rejects_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            REFERENCE.vector_summary([1.0, float("nan")], [0])

    def test_generation_eos_ids_are_exact_ordered_integers(self) -> None:
        self.assertEqual(
            REFERENCE.validated_generation_eos_token_ids(
                {"eos_token_id": [248_046, 248_044]}
            ),
            [248_046, 248_044],
        )
        for invalid in ([248_044, 248_046], [248_046], 248_044, [True, 248_044]):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "EOS IDs changed"):
                    REFERENCE.validated_generation_eos_token_ids(
                        {"eos_token_id": invalid}
                    )

    def test_pinned_prompt_is_a_varied_stable_three_word_cycle(self) -> None:
        """Pin the diagnostic-proven 113x-ULP fixture without using a trivial token loop."""

        self.assertEqual(REFERENCE.PROMPT, "one two three one two three")
        self.assertEqual(
            REFERENCE.INPUT_IDS,
            [588, 1_330, 2_250, 799, 1_330, 2_250],
        )
        self.assertEqual(REFERENCE.INPUT_IDS[1:3], REFERENCE.INPUT_IDS[4:6])
        self.assertNotEqual(
            REFERENCE.INPUT_IDS[0],
            REFERENCE.INPUT_IDS[3],
            "sentence-initial and space-prefixed `one` must remain distinct tokens",
        )
        self.assertGreater(len(set(REFERENCE.INPUT_IDS)), 1)

    def test_binary16_ulp_is_scale_aware_and_requires_exact_f16(self) -> None:
        self.assertEqual(REFERENCE.binary16_ulp(0.0), 2.0**-24)
        self.assertEqual(REFERENCE.binary16_ulp(1.0), 2.0**-10)
        self.assertEqual(REFERENCE.binary16_ulp(8.0), 2.0**-7)
        self.assertEqual(REFERENCE.binary16_ulp(7.99609375), 2.0**-8)
        with self.assertRaisesRegex(ValueError, "not exactly representable"):
            REFERENCE.binary16_ulp(0.1)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            REFERENCE.binary16_ulp(float("inf"))

    def test_greedy_stability_uses_binary16_perturbation_proof(self) -> None:
        decision = REFERENCE.greedy_decision_stability(
            step=2,
            top1_token_id=7,
            top2_token_id=6,
            top1_logit=8.046875,
            top2_logit=7.96484375,
            eos_token_ids=[8, 9],
        )
        self.assertEqual(decision["margin"], 0.08203125)
        self.assertEqual(decision["top1_f16_ulp"], 0.0078125)
        self.assertEqual(decision["top2_f16_ulp"], 0.00390625)
        self.assertEqual(decision["elementwise_epsilon"], 0.0078125)
        self.assertEqual(decision["margin_to_epsilon_ratio"], 10.5)
        self.assertTrue(decision["stable_under_budget"])
        self.assertEqual(
            REFERENCE.validate_decision_stability([decision]),
            (0.08203125, 10.5),
        )

    def test_greedy_stability_reports_all_unstable_steps_after_collection(self) -> None:
        unstable = [
            REFERENCE.greedy_decision_stability(
                step=step,
                top1_token_id=7,
                top2_token_id=6,
                top1_logit=16.0,
                top2_logit=runner_up,
                eos_token_ids=[8, 9],
            )
            for step, runner_up in ((1, 15.984375), (2, 15.96875))
        ]
        self.assertEqual(unstable[1]["margin_to_epsilon_ratio"], 2.0)
        self.assertTrue(all(not item["stable_under_budget"] for item in unstable))
        with self.assertRaisesRegex(RuntimeError, r"steps \[1, 2\].*minimum_margin="):
            REFERENCE.validate_decision_stability(unstable)

    def test_greedy_stability_rejects_non_finite_or_eos_decisions(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            REFERENCE.greedy_decision_stability(
                step=1,
                top1_token_id=7,
                top2_token_id=6,
                top1_logit=float("nan"),
                top2_logit=1.0,
                eos_token_ids=[8, 9],
            )
        with self.assertRaisesRegex(RuntimeError, "reached EOS token 8"):
            REFERENCE.greedy_decision_stability(
                step=3,
                top1_token_id=8,
                top2_token_id=6,
                top1_logit=2.0,
                top2_logit=1.0,
                eos_token_ids=[8, 9],
            )

    def test_strict_json_rejects_non_finite_numbers(self) -> None:
        self.assertEqual(
            REFERENCE.strict_json_bytes({"finite": 1.0}),
            b'{\n  "finite": 1.0\n}\n',
        )
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Out of range float values"):
                    REFERENCE.strict_json_bytes({"invalid": value})

    def test_artifact_digest_is_order_independent(self) -> None:
        first = {"relative_path": "a", "bytes": 1, "sha256": "0" * 64}
        second = {"relative_path": "b", "bytes": 2, "sha256": "f" * 64}
        self.assertEqual(
            REFERENCE.canonical_artifact_digest([first, second]),
            REFERENCE.canonical_artifact_digest([second, first]),
        )

    def test_installed_distribution_provenance_matches_prepared_environment(self) -> None:
        versions, provenance, manifest = REFERENCE.verify_packages()
        self.assertEqual(versions, REFERENCE.EXPECTED_PACKAGE_VERSIONS)
        self.assertEqual(
            manifest, REFERENCE.EXPECTED_DISTRIBUTION_MANIFEST_SHA256
        )
        self.assertEqual(set(provenance), set(REFERENCE.EXPECTED_PACKAGE_VERSIONS))
        for name, identity in provenance.items():
            with self.subTest(distribution=name):
                self.assertEqual(identity["version"], versions[name])
                self.assertEqual(
                    identity["metadata"],
                    REFERENCE.EXPECTED_DISTRIBUTION_FILES[name]["METADATA"],
                )
                self.assertEqual(
                    identity["record"],
                    REFERENCE.EXPECTED_DISTRIBUTION_FILES[name]["RECORD"],
                )

    def test_distribution_manifest_is_order_independent(self) -> None:
        records = {
            name: {
                "version": REFERENCE.EXPECTED_PACKAGE_VERSIONS[name],
                "metadata": files["METADATA"],
                "record": files["RECORD"],
            }
            for name, files in REFERENCE.EXPECTED_DISTRIBUTION_FILES.items()
        }
        reversed_records = dict(reversed(list(records.items())))
        self.assertEqual(
            REFERENCE.canonical_distribution_digest(records),
            REFERENCE.canonical_distribution_digest(reversed_records),
        )
        self.assertEqual(
            REFERENCE.canonical_distribution_digest(records),
            REFERENCE.EXPECTED_DISTRIBUTION_MANIFEST_SHA256,
        )

    def test_transformers_reference_sources_match_pinned_wheel(self) -> None:
        distribution = importlib.metadata.distribution("transformers")
        package_entries = [
            entry
            for entry in distribution.files or ()
            if str(entry) == "transformers/__init__.py"
        ]
        self.assertEqual(len(package_entries), 1)
        transformers_root = Path(distribution.locate_file(package_entries[0])).parent
        sources = REFERENCE.verify_reference_sources(transformers_root)
        self.assertEqual(len(sources), 8)
        self.assertEqual(
            {source["module"] for source in sources},
            set(REFERENCE.EXPECTED_REFERENCE_SOURCES),
        )
        for source in sources:
            with self.subTest(module=source["module"]):
                expected = REFERENCE.EXPECTED_REFERENCE_SOURCES[source["module"]]
                self.assertEqual(source["relative_path"], expected["relative_path"])
                self.assertEqual(source["bytes"], expected["bytes"])
                self.assertEqual(source["sha256"], expected["sha256"])
                self.assertTrue(Path(source["path"]).is_absolute())

    def test_mps_allocator_warmup_override_is_exact_and_recorded(self) -> None:
        owner, original = fake_modeling_utils()
        loaded_sentinel = object()

        def loader() -> object:
            result = owner.caching_allocator_warmup(
                FakeQwenModel(), {"first": "mps", "second": "mps:0"}, None
            )
            self.assertIsNone(result)
            return loaded_sentinel

        loaded, provenance = REFERENCE.load_with_verified_mps_allocator_warmup_noop(
            owner, FakeTorch, FakeQwenModel, loader
        )
        self.assertIs(loaded, loaded_sentinel)
        self.assertIs(owner.caching_allocator_warmup, original)
        self.assertEqual(
            provenance,
            {
                "target": REFERENCE.ALLOCATOR_WARMUP_TARGET,
                "policy": REFERENCE.ALLOCATOR_WARMUP_POLICY,
                "scope": REFERENCE.ALLOCATOR_WARMUP_SCOPE,
                "reason": REFERENCE.ALLOCATOR_WARMUP_REASON,
                "upstream_source_module": REFERENCE.ALLOCATOR_WARMUP_SOURCE_MODULE,
                "call_count": 1,
                "expanded_device_map_entries": 2,
                "expanded_device_types": ["mps"],
                "hf_quantizer_present": False,
                "restored": True,
            },
        )

    def test_mps_allocator_warmup_override_requires_exactly_one_call(self) -> None:
        owner, original = fake_modeling_utils()
        with self.assertRaisesRegex(RuntimeError, "invocation count changed"):
            REFERENCE.load_with_verified_mps_allocator_warmup_noop(
                owner, FakeTorch, FakeQwenModel, lambda: "loaded"
            )
        self.assertIs(owner.caching_allocator_warmup, original)

        def twice() -> None:
            owner.caching_allocator_warmup(FakeQwenModel(), {"": "mps"}, None)
            owner.caching_allocator_warmup(FakeQwenModel(), {"": "mps"}, None)

        with self.assertRaisesRegex(RuntimeError, "invoked more than once"):
            REFERENCE.load_with_verified_mps_allocator_warmup_noop(
                owner, FakeTorch, FakeQwenModel, twice
            )
        self.assertIs(owner.caching_allocator_warmup, original)

    def test_mps_allocator_warmup_override_rejects_non_mps_maps(self) -> None:
        owner, original = fake_modeling_utils()

        def mixed_devices() -> None:
            owner.caching_allocator_warmup(
                FakeQwenModel(), {"first": "mps", "second": "cpu"}, None
            )

        with self.assertRaisesRegex(RuntimeError, "exclusively MPS"):
            REFERENCE.load_with_verified_mps_allocator_warmup_noop(
                owner, FakeTorch, FakeQwenModel, mixed_devices
            )
        self.assertIs(owner.caching_allocator_warmup, original)

    def test_mps_allocator_warmup_override_restores_after_loader_failure(self) -> None:
        owner, original = fake_modeling_utils()

        def failing_loader() -> None:
            owner.caching_allocator_warmup(FakeQwenModel(), {"": "mps"}, None)
            raise ValueError("injected model-load failure")

        with self.assertRaisesRegex(ValueError, "injected model-load failure"):
            REFERENCE.load_with_verified_mps_allocator_warmup_noop(
                owner, FakeTorch, FakeQwenModel, failing_loader
            )
        self.assertIs(owner.caching_allocator_warmup, original)

    def test_mps_allocator_warmup_override_rejects_quantized_or_wrong_model(self) -> None:
        invalid_calls = [
            (object(), {"": "mps"}, None, "unexpected model class"),
            (FakeQwenModel(), {"": "mps"}, object(), "forbidden for quantized"),
            (FakeQwenModel(), {}, None, "nonempty expanded device map"),
        ]
        for model, device_map, quantizer, message in invalid_calls:
            with self.subTest(message=message):
                owner, original = fake_modeling_utils()

                def invalid_loader() -> None:
                    owner.caching_allocator_warmup(model, device_map, quantizer)

                with self.assertRaisesRegex(RuntimeError, message):
                    REFERENCE.load_with_verified_mps_allocator_warmup_noop(
                        owner, FakeTorch, FakeQwenModel, invalid_loader
                    )
                self.assertIs(owner.caching_allocator_warmup, original)

    def test_write_create_only_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.bin"
            REFERENCE.write_create_only(output, b"first")
            with self.assertRaises(FileExistsError):
                REFERENCE.write_create_only(output, b"second")
            self.assertEqual(output.read_bytes(), b"first")

    def test_path_lexists_detects_dangling_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            link = root / "dangling"
            link.symlink_to(root / "missing")
            self.assertTrue(REFERENCE.path_lexists(link))
            self.assertFalse(REFERENCE.path_lexists(root / "absent"))

    def test_atomic_directory_rename_never_replaces_empty_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "payload").write_bytes(b"evidence")
            with self.assertRaises(FileExistsError):
                REFERENCE.rename_directory_create_only(source, destination)
            self.assertEqual((source / "payload").read_bytes(), b"evidence")
            self.assertTrue(destination.is_dir())

    def test_bundle_publication_is_atomic_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = REFERENCE.publish_reference_bundle(root, b"raw", b"json")
            self.assertEqual(bundle, root / REFERENCE.BUNDLE_DIRNAME)
            self.assertEqual((bundle / REFERENCE.RAW_FILENAME).read_bytes(), b"raw")
            self.assertEqual((bundle / REFERENCE.JSON_FILENAME).read_bytes(), b"json")
            self.assertEqual(
                list(root.glob(f".{REFERENCE.BUNDLE_DIRNAME}.staging-*")), []
            )
            with self.assertRaises(FileExistsError):
                REFERENCE.publish_reference_bundle(root, b"new", b"new")
            self.assertEqual((bundle / REFERENCE.RAW_FILENAME).read_bytes(), b"raw")

    def test_failed_bundle_write_leaves_recoverable_hidden_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_write = REFERENCE.write_create_only

            def fail_json(path: Path, payload: bytes) -> None:
                if path.name == REFERENCE.JSON_FILENAME:
                    raise OSError("injected JSON write failure")
                original_write(path, payload)

            with mock.patch.object(REFERENCE, "write_create_only", side_effect=fail_json):
                with self.assertRaisesRegex(OSError, "injected JSON write failure") as raised:
                    REFERENCE.publish_reference_bundle(root, b"raw", b"json")
            self.assertFalse((root / REFERENCE.BUNDLE_DIRNAME).exists())
            staging = list(root.glob(f".{REFERENCE.BUNDLE_DIRNAME}.staging-*"))
            self.assertEqual(len(staging), 1)
            self.assertEqual((staging[0] / REFERENCE.RAW_FILENAME).read_bytes(), b"raw")
            self.assertTrue(
                any(
                    "recoverable reference staging directory" in note
                    for note in raised.exception.__notes__
                )
            )

    def test_external_path_preflight_rejects_existing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            output = root / "output"
            model.mkdir()
            output.mkdir()
            resolved_model, resolved_output, bundle = REFERENCE.ensure_external_paths(
                model, output
            )
            self.assertEqual(resolved_model, model.resolve())
            self.assertEqual(resolved_output, output.resolve())
            self.assertEqual(bundle, output.resolve() / REFERENCE.BUNDLE_DIRNAME)
            bundle.mkdir()
            with self.assertRaises(FileExistsError):
                REFERENCE.ensure_external_paths(model, output)
            self.assertTrue(bundle.is_dir())

    def test_cli_is_mps_only_and_has_no_device_switch(self) -> None:
        arguments = REFERENCE.parse_arguments(
            ["--model-dir", "model", "--output-dir", "output"]
        )
        self.assertFalse(hasattr(arguments, "device"))
        with mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                REFERENCE.parse_arguments(
                    [
                        "--model-dir",
                        "model",
                        "--output-dir",
                        "output",
                        "--device",
                        "cuda",
                    ]
                )

    def test_checked_duration_rejects_negative_or_non_finite_values(self) -> None:
        self.assertEqual(REFERENCE.checked_duration("test", 0.0), 0.0)
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "duration is invalid"):
                    REFERENCE.checked_duration("test", value)


if __name__ == "__main__":
    unittest.main()
