"""Pure tests for the opt-in Qwen3.5-0.8B reference generator."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "generate_qwen35_08b_reference.py"
SPEC = importlib.util.spec_from_file_location("generate_qwen35_08b_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Qwen35ReferenceGeneratorTests(unittest.TestCase):
    """Exercise bounded helpers without importing Torch or Transformers."""

    def test_pinned_profile_and_continuation_are_complete(self) -> None:
        self.assertEqual(MODULE.REPOSITORY, "Qwen/Qwen3.5-0.8B")
        self.assertEqual(
            MODULE.REVISION, "2fc06364715b967f1860aea9cf38778875588b17"
        )
        self.assertEqual(MODULE.INPUT_IDS, [9_419])
        self.assertEqual(len(MODULE.EXPECTED_GREEDY_TOKEN_IDS), MODULE.GREEDY_TOKENS)
        self.assertFalse(
            set(MODULE.EXPECTED_GREEDY_TOKEN_IDS) & set(MODULE.EXPECTED_EOS_TOKEN_IDS)
        )
        self.assertEqual(len(MODULE.EXPECTED_ARTIFACTS), 6)

    def test_json_parser_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            MODULE.parse_json_bytes(b'{"value": 1, "value": 2}', "fixture")
        self.assertEqual(
            MODULE.parse_json_bytes(b'{"value": [1, 2]}', "fixture"),
            {"value": [1, 2]},
        )

    def test_artifact_digest_is_order_independent_and_sensitive(self) -> None:
        left = [
            {"relative_path": "b", "bytes": 2, "sha256": "b" * 64},
            {"relative_path": "a", "bytes": 1, "sha256": "a" * 64},
        ]
        right = list(reversed(left))
        self.assertEqual(
            MODULE.canonical_artifact_digest(left),
            MODULE.canonical_artifact_digest(right),
        )
        changed = [dict(item) for item in left]
        changed[0]["bytes"] = 3
        self.assertNotEqual(
            MODULE.canonical_artifact_digest(left),
            MODULE.canonical_artifact_digest(changed),
        )

    def test_payload_record_validates_shape_and_digest(self) -> None:
        record = MODULE.payload_record("row.f32le", b"\0" * 24, [2, 3])
        self.assertEqual(record["shape"], [2, 3])
        self.assertEqual(record["elements"], 6)
        self.assertEqual(record["bytes"], 24)
        self.assertEqual(len(record["sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "does not match shape"):
            MODULE.payload_record("row.f32le", b"\0" * 20, [2, 3])

    def test_bundle_publication_is_complete_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            files = {
                MODULE.PROMPT_LOGITS_FILENAME: b"prompt",
                MODULE.CACHED_LOGITS_FILENAME: b"cached",
                MODULE.TEACHER_LOGITS_FILENAME: b"teacher",
                MODULE.JSON_FILENAME: b"{}\n",
            }
            bundle = MODULE.publish_bundle(output, files)
            self.assertEqual(bundle.name, MODULE.BUNDLE_DIRNAME)
            self.assertEqual(
                {path.name for path in bundle.iterdir()},
                set(files),
            )
            for name, payload in files.items():
                self.assertEqual((bundle / name).read_bytes(), payload)
            with self.assertRaises(FileExistsError):
                MODULE.publish_bundle(output, files)

    def test_bundle_publication_rejects_incomplete_file_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "file set changed"):
                MODULE.publish_bundle(Path(temporary), {MODULE.JSON_FILENAME: b"{}"})


if __name__ == "__main__":
    unittest.main()
