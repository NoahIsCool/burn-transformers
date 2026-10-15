"""Pure tests for the generic causal-LM reference generator (no PyTorch import)."""

from __future__ import annotations

import importlib.util
import math
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "generate_causal_lm_reference.py"
SPEC = importlib.util.spec_from_file_location("generate_causal_lm_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class FakeTensor:
    """Stands in for a contiguous CPU tensor whose `numpy()` returns an array."""

    def __init__(self, array):
        """Wraps an array."""

        self.array = array

    def contiguous(self) -> "FakeTensor":
        """Returns itself, as a contiguous tensor would."""

        return self

    def numpy(self):
        """Returns the wrapped array."""

        return self.array


class ReferenceGeneratorTests(unittest.TestCase):
    """Check the format helpers, argument handling, and safe publication."""

    def test_offline_controls_are_installed_at_import(self) -> None:
        """Importing the script switches every library to offline mode."""

        import os

        for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            self.assertEqual(os.environ[variable], "1")

    def test_f32le_round_trips_and_is_little_endian(self) -> None:
        """Values survive encoding exactly and use the documented byte order."""

        values = [0.0, -1.5, 3.25, 2.0**-100, float("inf")]
        payload = REFERENCE.encode_f32le(values)
        self.assertEqual(payload[:8], struct.pack("<2f", 0.0, -1.5))
        self.assertEqual(REFERENCE.decode_f32le(payload), values)
        with self.assertRaises(ValueError):
            REFERENCE.decode_f32le(b"\x00\x00\x00")

    def test_tensor_serialization_matches_the_reference_encoding(self) -> None:
        """The fast tensor path writes the same bytes as the pure encoder."""

        try:
            import numpy
        except ImportError:  # pragma: no cover - numpy ships with PyTorch.
            self.skipTest("numpy is unavailable")
        array = numpy.array([[1.0, -2.0], [0.5, 7.25]], dtype=numpy.float32)
        self.assertEqual(
            REFERENCE.tensor_f32le(FakeTensor(array)),
            REFERENCE.encode_f32le([1.0, -2.0, 0.5, 7.25]),
        )

    def test_top_two_margin_prefers_the_lowest_index_on_ties(self) -> None:
        """Margins are the gap to the runner-up; ties have a zero margin."""

        self.assertEqual(REFERENCE.top_two_margin([0.0, 3.0, 1.0]), (1, 2.0))
        token, margin = REFERENCE.top_two_margin([2.0, 5.0, 5.0])
        self.assertEqual((token, margin), (1, 0.0))
        self.assertTrue(math.isfinite(REFERENCE.top_two_margin([-1.0, -2.0])[1]))
        with self.assertRaises(ValueError):
            REFERENCE.top_two_margin([1.0])

    def test_arguments_default_to_the_fixed_prompts(self) -> None:
        """Without prompts the three fixed prompts and 32 greedy tokens are used."""

        arguments = REFERENCE.parse_arguments(["--model-dir", "m", "--output", "o"])
        self.assertEqual(arguments.prompts, list(REFERENCE.DEFAULT_PROMPTS))
        self.assertEqual(arguments.greedy_tokens, 32)
        self.assertTrue(arguments.add_special_tokens)
        custom = REFERENCE.parse_arguments(
            ["--model-dir", "m", "--output", "o", "--prompt", "a", "--prompt", "b",
             "--greedy-tokens", "4", "--no-special-tokens", "--revision", "abc"]
        )
        self.assertEqual(custom.prompts, ["a", "b"])
        self.assertEqual((custom.greedy_tokens, custom.add_special_tokens, custom.revision), (4, False, "abc"))
        with self.assertRaises(SystemExit):
            REFERENCE.parse_arguments(["--model-dir", "m", "--output", "o", "--greedy-tokens", "0"])

    def test_dtype_defaults_to_float32_and_accepts_bfloat16(self) -> None:
        """The evaluation dtype is F32 unless BF16 is requested; others are refused."""

        base = ["--model-dir", "m", "--output", "o"]
        self.assertEqual(REFERENCE.parse_arguments(base).dtype, "float32")
        self.assertEqual(REFERENCE.parse_arguments(base + ["--dtype", "bfloat16"]).dtype, "bfloat16")
        self.assertEqual(REFERENCE.DTYPES, ("float32", "bfloat16"))
        with self.assertRaises(SystemExit):
            REFERENCE.parse_arguments(base + ["--dtype", "float16"])
        with self.assertRaises(ValueError):
            REFERENCE.load_model(Path("m"), "float16")

    def test_publication_never_replaces_a_bundle(self) -> None:
        """A finished staging directory is renamed once; an existing bundle is kept."""

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            staging = root / ".bundle.tmp"
            staging.mkdir()
            (staging / "reference.json").write_text("{}")
            destination = root / "bundle"
            REFERENCE.publish(staging, destination)
            self.assertTrue((destination / "reference.json").is_file())
            again = root / ".bundle.again"
            again.mkdir()
            with self.assertRaises(FileExistsError):
                REFERENCE.publish(again, destination)
            with self.assertRaises(FileExistsError):
                REFERENCE.main(["--model-dir", str(root), "--output", str(destination)])

    def test_failed_generation_removes_its_staging_directory(self) -> None:
        """A failure before publication leaves no hidden staging directory behind."""

        from unittest import mock

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            destination = root / "bundle"

            def fail(arguments, staging):
                (staging / "logits-0.f32le").write_bytes(b"partial")
                raise MemoryError("simulated out-of-memory")

            with mock.patch.object(REFERENCE, "build_bundle", side_effect=fail):
                with self.assertRaises(MemoryError):
                    REFERENCE.main(["--model-dir", str(root), "--output", str(destination)])
            self.assertEqual(list(root.iterdir()), [])

    def test_weight_files_are_listed_or_reported_missing(self) -> None:
        """Weight discovery is sorted and refuses an empty checkpoint."""

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with self.assertRaises(FileNotFoundError):
                REFERENCE.weight_files(root)
            for name in ("b.safetensors", "a.safetensors", "c.bin"):
                (root / name).write_bytes(b"x")
            self.assertEqual([path.name for path in REFERENCE.weight_files(root)], ["a.safetensors", "b.safetensors"])
            self.assertEqual(
                REFERENCE.sha256_file(root / "a.safetensors"),
                "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881",
            )


if __name__ == "__main__":
    unittest.main()
