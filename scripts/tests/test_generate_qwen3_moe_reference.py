"""Pure tests for the pinned Qwen3-30B-A3B reference workflow."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "generate_qwen3_moe_reference.py"
SPEC = importlib.util.spec_from_file_location("generate_qwen3_moe_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class FakeTorch:
    """Minimal device facade for allocator-policy tests."""

    @staticmethod
    def device(value: object) -> object:
        """Parse the device type from a string assignment."""

        if not isinstance(value, str) or not value:
            raise TypeError("invalid fake device")
        return types.SimpleNamespace(type=value.partition(":")[0])


class FakeQwen3Moe:
    """Exact model-class sentinel accepted by the allocator override."""


class FakeRecordEntry:
    """Small importlib.metadata.PackagePath-shaped RECORD entry."""

    def __init__(self, relative: str, payload: bytes | None) -> None:
        self.relative = relative
        self.name = Path(relative).name
        self.parent = Path(relative).parent
        self.size = None if payload is None else len(payload)
        if payload is None:
            self.hash = None
        else:
            encoded = REFERENCE.base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
            self.hash = types.SimpleNamespace(
                mode="sha256", value=encoded.decode("ascii").rstrip("=")
            )

    def __str__(self) -> str:
        return self.relative


def fake_modeling_utils() -> tuple[object, object]:
    """Return an upstream-shaped owner and its original warmup callable."""

    def caching_allocator_warmup(
        _model: object, _device_map: object, _quantizer: object
    ) -> None:
        raise AssertionError("original warmup must be intercepted")

    caching_allocator_warmup.__module__ = "transformers.modeling_utils"
    owner = types.SimpleNamespace(
        __name__="transformers.modeling_utils",
        caching_allocator_warmup=caching_allocator_warmup,
    )
    return owner, caching_allocator_warmup


class ReferenceHelperTests(unittest.TestCase):
    """Exercise every safety-critical helper without importing the ML stack."""

    def test_import_is_lightweight_and_installs_offline_controls(self) -> None:
        for name, value in {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "USE_HUB_KERNELS": "0",
        }.items():
            self.assertEqual(REFERENCE.os.environ[name], value)
        syntax = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        top_level_imports = {
            alias.name.partition(".")[0]
            for node in syntax.body
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").partition(".")[0]
            for node in syntax.body
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("torch", top_level_imports)
        self.assertNotIn("transformers", top_level_imports)

    def test_shared_helper_is_bound_before_execution(self) -> None:
        payload = REFERENCE.COMMON_HELPER_PATH.read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(), REFERENCE.COMMON_HELPER_SHA256
        )
        self.assertEqual(payload, REFERENCE._COMMON_SOURCE_BYTES)
        self.assertNotIn("exec_module", SCRIPT.read_text(encoding="utf-8"))

    def test_pinned_identity_and_prompt_are_exact(self) -> None:
        self.assertEqual(REFERENCE.REPOSITORY, "Qwen/Qwen3-30B-A3B")
        self.assertEqual(
            REFERENCE.REVISION, "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
        )
        self.assertEqual(REFERENCE.REFERENCE_CLASS, "transformers.Qwen3MoeForCausalLM")
        self.assertEqual(REFERENCE.REFERENCE_SOURCE_TAG, "v5.2.0")
        self.assertEqual(
            REFERENCE.REFERENCE_SOURCE_COMMIT,
            "7d9754a05193eb79b1d86aa744b622b8068008cd",
        )
        self.assertEqual(REFERENCE.VOCAB_SIZE, 151_936)
        self.assertEqual(REFERENCE.EXPECTED_PARAMETER_COUNT, 30_532_122_624)
        self.assertEqual(REFERENCE.PROMPT, "Hello world, cached decoding.")
        self.assertEqual(
            REFERENCE.INPUT_IDS, [9_707, 1_879, 11, 20_579, 47_116, 13]
        )

    def test_generation_eos_ids_require_exact_ordered_integers(self) -> None:
        self.assertEqual(
            REFERENCE.validated_generation_eos_token_ids(
                {"eos_token_id": [151_645, 151_643]}
            ),
            [151_645, 151_643],
        )
        for value in ([151_643, 151_645], [151_645], 151_645, [True, 151_643]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "EOS IDs changed"):
                    REFERENCE.validated_generation_eos_token_ids({"eos_token_id": value})

    def test_greedy_stability_uses_both_candidates_f16_spacing(self) -> None:
        decision = REFERENCE.greedy_decision_stability(
            step=7,
            top1_token_id=12,
            top2_token_id=11,
            top1_logit=8.046875,
            top2_logit=7.96484375,
            eos_token_ids=[20, 21],
        )
        self.assertEqual(decision["margin"], 0.08203125)
        self.assertEqual(decision["top1_f16_ulp"], 0.0078125)
        self.assertEqual(decision["top2_f16_ulp"], 0.00390625)
        self.assertEqual(decision["margin_to_epsilon_ratio"], 10.5)
        self.assertTrue(decision["stable_under_budget"])

    def test_greedy_stability_rejects_eos_ties_order_and_nonfinite(self) -> None:
        cases = [
            ({"top1_token_id": 20, "top2_token_id": 1}, "reached EOS"),
            ({"top1_token_id": 1, "top2_token_id": 1}, "equal"),
            ({"top1_logit": 0.0, "top2_logit": 1.0}, "out of order"),
            ({"top1_logit": float("nan"), "top2_logit": 1.0}, "non-finite"),
        ]
        base = {
            "step": 0,
            "top1_token_id": 2,
            "top2_token_id": 1,
            "top1_logit": 2.0,
            "top2_logit": 1.0,
            "eos_token_ids": [20, 21],
        }
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    REFERENCE.greedy_decision_stability(**(base | overrides))

    def test_stability_sequence_requires_32_stable_decisions(self) -> None:
        decision = REFERENCE.greedy_decision_stability(
            step=0,
            top1_token_id=2,
            top2_token_id=1,
            top1_logit=2.0,
            top2_logit=1.0,
            eos_token_ids=[20, 21],
        )
        decisions = [dict(decision, step=step) for step in range(32)]
        self.assertEqual(REFERENCE.validate_decision_stability(decisions), (1.0, 512.0))
        with self.assertRaisesRegex(RuntimeError, "expected 32"):
            REFERENCE.validate_decision_stability(decisions[:-1])
        decisions[3]["stable_under_budget"] = False
        with self.assertRaisesRegex(RuntimeError, r"steps \[3\]"):
            REFERENCE.validate_decision_stability(decisions)

    def test_optional_kernel_distributions_must_be_absent(self) -> None:
        with mock.patch.object(
            REFERENCE.importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ) as version:
            self.assertEqual(
                REFERENCE.verify_absent_optional_distributions(),
                ["flash-attn", "kernels", "torchao"],
            )
            self.assertEqual(version.call_count, 3)
        with mock.patch.object(
            REFERENCE.importlib.metadata,
            "version",
            side_effect=[importlib.metadata.PackageNotFoundError, "1.0.0"],
        ):
            with self.assertRaisesRegex(RuntimeError, "kernels.*must be absent"):
                REFERENCE.verify_absent_optional_distributions()

    def test_official_source_manifest_matches_installed_pinned_wheel(self) -> None:
        root = REFERENCE.transformers_package_root()
        sources, identities = REFERENCE.verify_reference_sources(root)
        self.assertEqual(len(sources), len(REFERENCE.EXPECTED_REFERENCE_SOURCES))
        self.assertEqual(set(identities), set(REFERENCE.EXPECTED_REFERENCE_SOURCES))
        self.assertEqual(
            {item["module"] for item in sources},
            set(REFERENCE.EXPECTED_REFERENCE_SOURCES),
        )

    def test_source_verification_rejects_changed_bytes_and_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_bytes(b"official")
            expected = {
                "transformers.model": {
                    "relative_path": "model.py",
                    "bytes": 8,
                    "sha256": hashlib.sha256(b"official").hexdigest(),
                }
            }
            with mock.patch.object(REFERENCE, "EXPECTED_REFERENCE_SOURCES", expected):
                self.assertEqual(
                    REFERENCE.verify_reference_sources(root)[0][0]["bytes"], 8
                )
                (root / "model.py").write_bytes(b"changed!")
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    REFERENCE.verify_reference_sources(root)
            with mock.patch.object(
                REFERENCE,
                "EXPECTED_REFERENCE_SOURCES",
                {"bad": {"relative_path": "../bad.py", "bytes": 0, "sha256": "0"}},
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid pinned source path"):
                    REFERENCE.verify_reference_sources(root)

    def test_checkpoint_preflight_validates_config_index_hashes_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "architectures": ["Qwen3MoeForCausalLM"],
                "attention_bias": False,
                "attention_dropout": 0.0,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2048,
                "max_position_embeddings": 40960,
                "mlp_only_layers": [],
                "model_type": "qwen3_moe",
                "moe_intermediate_size": 768,
                "norm_topk_prob": True,
                "num_attention_heads": 32,
                "num_experts": 128,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 48,
                "num_key_value_heads": 4,
                "rope_theta": 1_000_000.0,
                "tie_word_embeddings": False,
                "torch_dtype": "bfloat16",
                "use_cache": True,
                "use_sliding_window": False,
                "vocab_size": 151936,
            }
            payloads = {
                "config.json": json.dumps(config).encode(),
                "generation_config.json": json.dumps(
                    {"eos_token_id": [151645, 151643]}
                ).encode(),
                "model.safetensors.index.json": json.dumps(
                    {
                        "metadata": {"total_size": 4},
                        "weight_map": {"a": "shard-a", "b": "shard-b"},
                    }
                ).encode(),
                "shard-a": b"a",
                "shard-b": b"b",
            }
            for name, payload in payloads.items():
                (root / name).write_bytes(payload)
            controls = {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in payloads.items()
                if name.endswith(".json")
            }
            shards = {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in payloads.items()
                if name.startswith("shard-")
            }
            with (
                mock.patch.object(REFERENCE, "EXPECTED_CONTROL_HASHES", controls),
                mock.patch.object(REFERENCE, "EXPECTED_SHARD_HASHES", shards),
                mock.patch.object(REFERENCE, "EXPECTED_PHYSICAL_TENSORS", 2),
                mock.patch.object(REFERENCE, "CHECKPOINT_TENSOR_BYTES", 4),
            ):
                artifacts, byte_count, digest, eos, identities = (
                    REFERENCE.verify_checkpoint(root)
                )
                self.assertEqual(len(artifacts), 5)
                self.assertEqual(set(identities), set(payloads))
                self.assertEqual(byte_count, 4)
                self.assertEqual(len(digest), 64)
                self.assertEqual(eos, [151645, 151643])
                config["num_experts"] = 127
                changed = json.dumps(config).encode()
                (root / "config.json").write_bytes(changed)
                controls["config.json"] = hashlib.sha256(changed).hexdigest()
                with self.assertRaisesRegex(ValueError, "num_experts.*changed"):
                    REFERENCE.verify_checkpoint(root)

    def test_authenticated_file_rejects_links_and_detects_identity_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"trusted")
            digest = hashlib.sha256(b"trusted").hexdigest()
            record, identity, payload = REFERENCE.authenticated_file(
                source,
                expected_sha256=digest,
                expected_bytes=7,
                retain_payload=True,
            )
            self.assertEqual(record, {"bytes": 7, "sha256": digest})
            self.assertEqual(payload, b"trusted")
            source.unlink()
            source.write_bytes(b"trusted")
            _, replacement_identity, _ = REFERENCE.authenticated_file(
                source, expected_sha256=digest, expected_bytes=7
            )
            self.assertNotEqual(identity, replacement_identity)
            link = root / "link"
            link.symlink_to(source)
            with self.assertRaises(OSError):
                REFERENCE.authenticated_file(link, expected_sha256=digest)

    def test_authenticated_import_prefix_is_empty_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            package = output / "private_package"
            package.mkdir()
            origin = package / "__init__.py"
            origin.write_text("# source\n", encoding="utf-8")
            previous = (REFERENCE.sys.dont_write_bytecode, REFERENCE.sys.pycache_prefix)
            with mock.patch.object(
                REFERENCE.importlib.util,
                "find_spec",
                return_value=types.SimpleNamespace(origin=str(origin)),
            ):
                cache, resolutions, saved = REFERENCE.prepare_authenticated_imports(
                    output, {"private_package": package.resolve()}
                )
            self.assertEqual(resolutions, {"private_package": str(origin.resolve())})
            self.assertTrue(REFERENCE.sys.dont_write_bytecode)
            policy = REFERENCE.finalize_authenticated_imports(cache, saved)
            self.assertTrue(policy["private_cache_remained_empty"])
            self.assertEqual(
                (REFERENCE.sys.dont_write_bytecode, REFERENCE.sys.pycache_prefix),
                previous,
            )

    def test_every_hashed_wheel_record_payload_is_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = "package/module.py"
            payload = b"executed source\n"
            path = root / relative
            path.parent.mkdir()
            path.write_bytes(payload)
            data_entry = FakeRecordEntry(relative, payload)
            record_entry = FakeRecordEntry("package-1.dist-info/RECORD", None)
            distribution = types.SimpleNamespace(
                files=[data_entry, record_entry],
                locate_file=lambda entry: root / str(entry),
            )
            canonical = (
                f"{relative}\0{len(payload)}\0{hashlib.sha256(payload).hexdigest()}\n"
            ).encode()
            expected = {
                "package": (
                    1,
                    len(payload),
                    0,
                    hashlib.sha256(canonical).hexdigest(),
                )
            }
            with (
                mock.patch.object(
                    REFERENCE.COMMON, "EXPECTED_PACKAGE_VERSIONS", {"package": "1"}
                ),
                mock.patch.object(REFERENCE, "EXPECTED_DISTRIBUTION_PAYLOADS", expected),
                mock.patch.object(
                    REFERENCE.importlib.metadata,
                    "distribution",
                    return_value=distribution,
                ),
            ):
                summaries, identities = REFERENCE.verify_distribution_payloads()
                self.assertEqual(summaries["package"]["verified_files"], 1)
                self.assertEqual(set(identities), {("package", relative)})
                path.write_bytes(b"changed source\n")
                with self.assertRaisesRegex(RuntimeError, "bytes|SHA-256"):
                    REFERENCE.verify_distribution_payloads()

    def test_teacher_top1_summary_survives_tensor_cleanup(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            '"all_position_top1_token_ids": teacher_all_top1,',
            source,
        )
        self.assertNotIn(
            '"all_position_top1_token_ids": teacher_all.argmax',
            source,
        )

    def test_output_path_must_be_external_absent_and_outside_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            output = root / "output"
            model.mkdir()
            output.mkdir()
            observed = REFERENCE.ensure_external_paths(model, output)
            self.assertEqual(
                observed[2], output.resolve() / REFERENCE.BUNDLE_DIRNAME
            )
            (output / REFERENCE.BUNDLE_DIRNAME).mkdir()
            with self.assertRaisesRegex(FileExistsError, "create-only"):
                REFERENCE.ensure_external_paths(model, output)
            nested = model / "nested"
            nested.mkdir()
            with self.assertRaisesRegex(ValueError, "outside the checkpoint"):
                REFERENCE.ensure_external_paths(model, nested)

    def test_bundle_publication_is_atomic_complete_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads = {
                REFERENCE.PROMPT_LOGITS_RAW_FILENAME: b"prompt",
                REFERENCE.CACHED_LOGITS_RAW_FILENAME: b"cached",
                REFERENCE.TEACHER_LOGITS_RAW_FILENAME: b"teacher",
                REFERENCE.JSON_FILENAME: b"json",
            }
            bundle = REFERENCE.publish_reference_bundle(root, payloads)
            self.assertEqual({path.name for path in bundle.iterdir()}, set(payloads))
            for name, payload in payloads.items():
                self.assertEqual((bundle / name).read_bytes(), payload)
            with self.assertRaisesRegex(FileExistsError, "create-only"):
                REFERENCE.publish_reference_bundle(root, payloads)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "payload set changed"):
                REFERENCE.publish_reference_bundle(Path(directory), {"only": b"one"})

    def test_allocator_override_accepts_only_exact_unquantized_mps_call(self) -> None:
        owner, original = fake_modeling_utils()

        def loader() -> object:
            owner.caching_allocator_warmup(
                FakeQwen3Moe(), {"weight": "mps:0"}, None
            )
            return "loaded"

        loaded, record = REFERENCE.load_with_verified_mps_allocator_warmup_noop(
            owner, FakeTorch, FakeQwen3Moe, loader
        )
        self.assertEqual(loaded, "loaded")
        self.assertEqual(record["call_count"], 1)
        self.assertTrue(record["restored"])
        self.assertIs(owner.caching_allocator_warmup, original)

    def test_allocator_override_restores_after_loader_failure(self) -> None:
        owner, original = fake_modeling_utils()

        def loader() -> object:
            owner.caching_allocator_warmup(FakeQwen3Moe(), {"weight": "cpu"}, None)
            return "unreachable"

        with self.assertRaisesRegex(RuntimeError, "only MPS"):
            REFERENCE.load_with_verified_mps_allocator_warmup_noop(
                owner, FakeTorch, FakeQwen3Moe, loader
            )
        self.assertIs(owner.caching_allocator_warmup, original)

    def test_loading_diagnostics_are_empty_and_content_bound(self) -> None:
        summary = REFERENCE.loading_info_summary({})
        empty_digest = hashlib.sha256(b"").hexdigest()
        self.assertTrue(all(item == {"count": 0, "sha256": empty_digest} for item in summary.values()))
        with self.assertRaisesRegex(RuntimeError, "unexpected_keys"):
            REFERENCE.loading_info_summary({"unexpected_keys": ["weight"]})

    def test_argument_parser_requires_both_local_paths(self) -> None:
        parsed = REFERENCE.parse_arguments(
            ["--model-dir", "/model", "--output-dir", "/output"]
        )
        self.assertEqual(parsed.model_dir, Path("/model"))
        self.assertEqual(parsed.output_dir, Path("/output"))
        with self.assertRaises(SystemExit):
            REFERENCE.parse_arguments([])


if __name__ == "__main__":
    unittest.main()
