"""Pure safety tests for the FLAN-T5-small reference generator."""

from __future__ import annotations

import importlib.util
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "generate_flan_t5_small_reference.py"
SPEC = importlib.util.spec_from_file_location("generate_flan_t5_small_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


def safetensors_bytes(header: object, payload: bytes = b"") -> bytes:
    """Encode a minimal SafeTensors envelope for parser tests."""

    encoded = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(encoded)) + encoded + payload


class FlanSmallReferenceTests(unittest.TestCase):
    """Exercise fail-closed helpers without loading the real model."""

    def test_hash_and_strict_json_helpers(self) -> None:
        """Pin standard SHA-256 and deterministic finite JSON behavior."""

        self.assertEqual(
            REFERENCE.sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        self.assertEqual(REFERENCE.strict_json_bytes({"b": 2, "a": 1}), b'{\n  "a": 1,\n  "b": 2\n}\n')
        with self.assertRaisesRegex(ValueError, "Out of range"):
            REFERENCE.strict_json_bytes({"value": float("nan")})

    def test_json_parsers_reject_duplicate_keys(self) -> None:
        """Reject ambiguity in both control files and tensor headers."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                REFERENCE.json_without_duplicate_keys(path)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            REFERENCE.json_bytes_without_duplicate_keys(b'{"x":1,"x":2}', "header")

    def test_regular_file_rejects_traversal_and_symlinks(self) -> None:
        """Keep checkpoint reads under the authenticated model directory."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "target"
            target.write_bytes(b"x")
            self.assertEqual(REFERENCE.regular_file(root, "target"), target)
            for name in ("../target", "/target", "./target"):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "local basename"):
                        REFERENCE.regular_file(root, name)
            (root / "link").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "nonsymlink"):
                REFERENCE.regular_file(root, "link")

    def test_closed_artifact_inventory_and_hashes(self) -> None:
        """Fail on missing, unexpected, substituted, and non-directory entries."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            payload = b"checkpoint"
            (root / "model.safetensors").write_bytes(payload)
            (root / ".cache").mkdir()
            files = {"model.safetensors": (len(payload), REFERENCE.sha256_bytes(payload))}
            with (
                mock.patch.object(REFERENCE, "EXPECTED_FILES", files),
                mock.patch.object(REFERENCE, "ALLOWED_ROOT_DIRECTORIES", {".cache"}),
            ):
                records = REFERENCE.verify_artifacts(root)
                self.assertEqual(records[0]["relative_path"], "model.safetensors")
                (root / "unexpected").write_bytes(b"x")
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    REFERENCE.verify_artifacts(root)
                (root / "unexpected").unlink()
                (root / "model.safetensors").write_bytes(b"substitution")
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    REFERENCE.verify_artifacts(root)

    def test_canonical_record_digest_is_order_independent(self) -> None:
        """Ensure provenance identities do not depend on directory iteration."""

        first = {"relative_path": "a", "bytes": 1, "sha256": "0" * 64}
        second = {"relative_path": "b", "bytes": 2, "sha256": "f" * 64}
        self.assertEqual(
            REFERENCE.canonical_records_digest([first, second]),
            REFERENCE.canonical_records_digest([second, first]),
        )

    def test_header_parser_bounds_and_rejects_ambiguity(self) -> None:
        """Reject truncated, oversized, and duplicate-key tensor headers."""

        valid_header = {
            "x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "__metadata__": {"format": "pt"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.safetensors"
            valid.write_bytes(safetensors_bytes(valid_header, b"\0" * 8))
            observed, start, size = REFERENCE.read_safetensors_header(valid)
            self.assertEqual(observed, valid_header)
            self.assertEqual(start + 8, size)

            truncated = root / "truncated.safetensors"
            truncated.write_bytes(struct.pack("<Q", 100) + b"{}")
            with self.assertRaisesRegex(ValueError, "exceeds file size"):
                REFERENCE.read_safetensors_header(truncated)

            oversized = root / "oversized.safetensors"
            oversized.write_bytes(struct.pack("<Q", REFERENCE.MAX_HEADER_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "header length is invalid"):
                REFERENCE.read_safetensors_header(oversized)

            duplicate = root / "duplicate.safetensors"
            duplicate_payload = b'{"x":{},"x":{}}'
            duplicate.write_bytes(struct.pack("<Q", len(duplicate_payload)) + duplicate_payload)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                REFERENCE.read_safetensors_header(duplicate)

    def test_safetensors_structure_requires_gap_free_f32_payload(self) -> None:
        """Validate tensor count, dtype, range length, and physical contiguity."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = {
                "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
                "__metadata__": {"format": "pt"},
            }
            path = root / "model.safetensors"
            path.write_bytes(safetensors_bytes(header, b"\0" * 12))
            with (
                mock.patch.object(REFERENCE, "EXPECTED_PHYSICAL_TENSORS", 2),
                mock.patch.object(REFERENCE, "EXPECTED_CHECKPOINT_TENSOR_BYTES", 12),
            ):
                summary = REFERENCE.verify_safetensors(root)
                self.assertEqual(summary["tensors"], 2)
                self.assertEqual(summary["tensor_bytes"], 12)
                header["b"]["data_offsets"] = [8, 16]
                path.write_bytes(safetensors_bytes(header, b"\0" * 16))
                with self.assertRaisesRegex(ValueError, "begins at"):
                    REFERENCE.verify_safetensors(root)
                header["b"] = {"dtype": "F16", "shape": [2], "data_offsets": [4, 8]}
                path.write_bytes(safetensors_bytes(header, b"\0" * 8))
                with self.assertRaisesRegex(ValueError, "not F32"):
                    REFERENCE.verify_safetensors(root)

    def test_boundary_input_is_exact_and_hashable(self) -> None:
        """Pin the maximum declared input construction without a 512-item literal."""

        values = REFERENCE.boundary_input_ids()
        self.assertEqual(len(values), 512)
        self.assertEqual(values[:6], [71, 8, 57, 71, 8, 57])
        self.assertEqual(values[-1], 1)
        self.assertEqual(
            REFERENCE.u32le_sha256(values),
            "08e561703c88bc7e2d08db781798024bbd09c4be470ae2ab200fde44198cbfe8",
        )

    def test_top_k_is_finite_and_stably_tied(self) -> None:
        """Order equal logits by lower token ID and reject non-finite vectors."""

        self.assertEqual(
            REFERENCE.top_k([1.0, 3.0, 3.0, -1.0], 3),
            [
                {"token_id": 1, "logit": 3.0},
                {"token_id": 2, "logit": 3.0},
                {"token_id": 0, "logit": 1.0},
            ],
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            REFERENCE.top_k([1.0, float("nan")], 1)

    def test_tensor_wire_format_and_parity_are_strict(self) -> None:
        """Use little-endian F32 and detect localized high-cosine corruption."""

        import numpy
        import torch

        payload, values = REFERENCE.tensor_payload(
            torch, numpy, torch.tensor([[1.0, -2.5]], dtype=torch.float64), [1, 2]
        )
        self.assertEqual(payload, struct.pack("<ff", 1.0, -2.5))
        self.assertEqual(values, [1.0, -2.5])
        with self.assertRaisesRegex(RuntimeError, "shape changed"):
            REFERENCE.tensor_payload(torch, numpy, torch.ones(2), [1, 2])
        expected = torch.ones(10_000)
        close = expected.clone()
        close[0] += 1.0e-5
        self.assertEqual(REFERENCE.parity_summary(torch, close, expected)["failure_count_at_atol_rtol_1e_4"], 0)
        corrupted = expected.clone()
        corrupted[0] += 0.1
        self.assertGreater(REFERENCE.cosine(torch, corrupted, expected), 0.9999)
        with self.assertRaisesRegex(RuntimeError, "reference paths disagree"):
            REFERENCE.parity_summary(torch, corrupted, expected)

    def test_output_paths_and_atomic_publication_are_create_only(self) -> None:
        """Reject nested destinations and never replace a visible bundle."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            output = root / "output"
            nested = model / "nested"
            model.mkdir()
            output.mkdir()
            nested.mkdir()
            _, _, bundle = REFERENCE.ensure_paths(model, output)
            with self.assertRaisesRegex(ValueError, "outside the checkpoint"):
                REFERENCE.ensure_paths(model, nested)
            payloads = {name: name.encode() for name in REFERENCE.RAW_FILES.values()}
            payloads[REFERENCE.JSON_FILENAME] = b"{}\n"
            published = REFERENCE.publish(output, payloads)
            self.assertEqual(published.resolve(), bundle.resolve())
            self.assertEqual({entry.name for entry in published.iterdir()}, set(payloads))
            with self.assertRaises(FileExistsError):
                REFERENCE.publish(output, payloads)

    def test_post_rename_failure_reports_visible_bundle(self) -> None:
        """Do not mislabel an already-published directory as recoverable staging."""

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            payloads = {name: name.encode() for name in REFERENCE.RAW_FILES.values()}
            payloads[REFERENCE.JSON_FILENAME] = b"{}\n"
            with mock.patch.object(
                REFERENCE,
                "fsync_directory",
                side_effect=[None, OSError("output sync failed")],
            ):
                with self.assertRaisesRegex(OSError, "output sync failed") as caught:
                    REFERENCE.publish(output, payloads)
            bundle = output / REFERENCE.BUNDLE_DIRNAME
            self.assertTrue(bundle.is_dir())
            self.assertIn(f"published bundle: {bundle}", caught.exception.__notes__)

    def test_exact_package_versions_are_available(self) -> None:
        """Guard the inexpensive environment half of maintainer preflight."""

        records = REFERENCE.verify_packages()
        self.assertEqual(
            {record["name"]: record["version"] for record in records},
            REFERENCE.EXPECTED_PACKAGES,
        )


if __name__ == "__main__":
    unittest.main()
