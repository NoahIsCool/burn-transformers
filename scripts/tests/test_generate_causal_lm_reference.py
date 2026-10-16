"""Tests for the generic causal-LM reference generator.

`ReferenceGeneratorTests` is pure (no PyTorch import). `ModelLoadingTests`
needs PyTorch and Transformers and is skipped without them; it loads the
committed tiny gpt-oss fixture, so it runs offline in a few seconds.
"""

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

#: Whether the model-loading tests can run.
HAS_TORCH = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))
#: Tiny seeded gpt-oss checkpoint (F32 weights, YaRN RoPE) committed for the Rust goldens.
GPT_OSS_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tiny_goldens" / "gpt-oss"
#: Tiny Qwen2 checkpoint (QKV biases, untied head) for dense F64 evaluation.
QWEN2_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tiny_goldens" / "qwen2"


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

    def test_dtype_defaults_to_float32_and_accepts_float64_and_bfloat16(self) -> None:
        """The evaluation dtype is F32 unless F64 or BF16 is requested; others are refused."""

        base = ["--model-dir", "m", "--output", "o"]
        self.assertEqual(REFERENCE.parse_arguments(base).dtype, "float32")
        self.assertEqual(REFERENCE.parse_arguments(base + ["--dtype", "float64"]).dtype, "float64")
        self.assertEqual(REFERENCE.parse_arguments(base + ["--dtype", "bfloat16"]).dtype, "bfloat16")
        self.assertEqual(REFERENCE.DTYPES, ("float32", "float64", "bfloat16"))
        with self.assertRaises(SystemExit):
            REFERENCE.parse_arguments(base + ["--dtype", "float16"])
        with self.assertRaises(ValueError):
            REFERENCE.load_model(Path("m"), "float16")

    def test_truncation_keeps_the_first_layers(self) -> None:
        """Truncation cuts the depth and the layer schedule; impossible depths are refused."""

        from types import SimpleNamespace

        config = SimpleNamespace(num_hidden_layers=4, layer_types=["s", "f", "s", "f"])
        REFERENCE.truncate_config(config, 2)
        self.assertEqual((config.num_hidden_layers, config.layer_types), (2, ["s", "f"]))
        plain = SimpleNamespace(num_hidden_layers=3)
        REFERENCE.truncate_config(plain, 3)
        self.assertEqual(plain.num_hidden_layers, 3)
        for layers in (0, 4):
            with self.assertRaises(ValueError):
                REFERENCE.truncate_config(plain, layers)
        base = ["--model-dir", "m", "--output", "o"]
        self.assertIsNone(REFERENCE.parse_arguments(base).num_hidden_layers)
        self.assertEqual(REFERENCE.parse_arguments(base + ["--num-hidden-layers", "1"]).num_hidden_layers, 1)
        with self.assertRaises(SystemExit):
            REFERENCE.parse_arguments(base + ["--num-hidden-layers", "0"])

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


@unittest.skipUnless(HAS_TORCH, "needs PyTorch and Transformers")
class ModelLoadingTests(unittest.TestCase):
    """Check that loaded references compute what plain Transformers inference computes."""

    def test_parameter_cast_leaves_buffers_and_integers_alone(self) -> None:
        """Floating parameters change dtype exactly; buffers and integer parameters keep theirs."""

        import torch

        module = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
        original = module.weight.detach().clone()
        frequencies = torch.tensor([1.0, 1.0 / 3.0], dtype=torch.float32)
        module.register_buffer("inv_freq", frequencies.clone())
        module.count = torch.nn.Parameter(torch.tensor([3]), requires_grad=False)
        self.assertEqual(REFERENCE.cast_floating_parameters(module, torch.float32), 2)
        self.assertEqual((module.weight.dtype, module.bias.dtype), (torch.float32, torch.float32))
        self.assertTrue(torch.equal(module.weight, original.to(torch.float32)))
        self.assertEqual(module.inv_freq.dtype, torch.float32)
        self.assertTrue(torch.equal(module.inv_freq, frequencies))
        self.assertEqual(module.count.dtype, torch.int64)
        self.assertEqual(REFERENCE.cast_floating_parameters(module, torch.float32), 0)
        # Downcasting must not touch the F32 buffer either (`Module.to` would).
        self.assertEqual(REFERENCE.cast_floating_parameters(module, torch.bfloat16), 2)
        self.assertEqual(module.inv_freq.dtype, torch.float32)

    def test_bfloat16_references_keep_f32_rotary_tables(self) -> None:
        """A BF16 reference equals plain BF16 Transformers inference, rotary tables included.

        The tiny fixture uses YaRN; rounding its ``inv_freq`` to BF16 changes
        the logits by about 1.2e-2, so this exact comparison detects any
        buffer cast.
        """

        import torch
        import transformers

        model = REFERENCE.load_causal_lm(GPT_OSS_FIXTURE, "bfloat16")
        tables = {name: buffer.dtype for name, buffer in model.named_buffers() if name.endswith("inv_freq")}
        self.assertTrue(tables)
        self.assertEqual(set(tables.values()), {torch.float32}, tables)
        self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.bfloat16})
        plain = transformers.AutoModelForCausalLM.from_pretrained(
            GPT_OSS_FIXTURE, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
        ).eval()
        tokens = torch.tensor([[1, 5, 9, 14, 3, 27, 40, 8]])
        with torch.no_grad():
            self.assertTrue(torch.equal(model(input_ids=tokens).logits, plain(input_ids=tokens).logits))

    def test_float32_references_are_entirely_f32_and_truncate(self) -> None:
        """An F32 reference holds only F32 parameters and buffers; `layers` keeps the first blocks."""

        import torch

        model = REFERENCE.load_causal_lm(GPT_OSS_FIXTURE, "float32")
        self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.float32})
        floating = {buffer.dtype for buffer in model.buffers() if buffer.is_floating_point()}
        self.assertEqual(floating, {torch.float32})
        self.assertEqual(len(model.model.layers), 4)
        truncated = REFERENCE.load_causal_lm(GPT_OSS_FIXTURE, "float32", layers=2)
        self.assertEqual(len(truncated.model.layers), 2)
        self.assertEqual(truncated.config.layer_types, ["sliding_attention", "full_attention"])
        with self.assertRaises(ValueError):
            REFERENCE.load_causal_lm(GPT_OSS_FIXTURE, "float16")

    def test_float64_references_hold_f64_parameters_and_agree_with_f32(self) -> None:
        """An F64 reference casts every parameter to F64 and stays within F32 rounding of F32 logits."""

        import torch

        model = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float64")
        self.assertEqual({parameter.dtype for parameter in model.parameters()}, {torch.float64})
        single = REFERENCE.load_causal_lm(QWEN2_FIXTURE, "float32")
        tokens = [1, 5, 9, 14, 3, 27, 40, 8]
        double_logits = REFERENCE.prompt_logits(model, tokens)
        single_logits = REFERENCE.prompt_logits(single, tokens)
        # Both are written as F32; the tiny fixture's logits are O(1), so F32
        # accumulation error is far below this bound while a wrong dtype path
        # (for example BF16 weights) would exceed it.
        self.assertEqual(double_logits.dtype, torch.float32)
        self.assertLess(float((double_logits - single_logits).abs().max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
