"""Pure tests for the Nemotron-H precision-reference tool (no PyTorch or checkpoint)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "generate_nemotron_h_precision_reference.py"
SPEC = importlib.util.spec_from_file_location("generate_nemotron_h_precision_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PRECISION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRECISION)

#: Whether the PyTorch-backed tests can run.
HAS_TORCH = importlib.util.find_spec("torch") is not None


def rows_payload(rows: list[list[float]]) -> bytes:
    """Encodes rows as consecutive little-endian binary32 values."""

    return b"".join(struct.pack(f"<{len(row)}f", *row) for row in rows)


class PrecisionReferenceTests(unittest.TestCase):
    """Check the comparison arithmetic and the fail-closed oracle reader."""

    def test_offline_controls_are_installed_at_import(self) -> None:
        """Importing the tool forbids Hub access before any ML package loads."""

        import os

        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_cosine_accumulates_exactly_and_rejects_bad_input(self) -> None:
        """Cosine is exact on small integers and refuses zero or mismatched vectors."""

        self.assertEqual(PRECISION.cosine([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(PRECISION.cosine([1.0, 1.0], [1.0, 0.0]), 1 / math.sqrt(2), places=15)
        self.assertEqual(PRECISION.cosine([3.0, -4.0], [-3.0, 4.0]), -1.0)
        with self.assertRaises(ValueError):
            PRECISION.cosine([0.0, 0.0], [1.0, 0.0])
        with self.assertRaises(ValueError):
            PRECISION.cosine([1.0], [1.0, 0.0])

    def test_summary_reports_the_first_minimum_and_top1_agreement(self) -> None:
        """The minimum row is the first on ties, and top-1 uses the lowest index on ties."""

        rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.5, 0.5, 0.0]]
        truth = [[1.0, 0.0, 0.0], [0.0, 0.5, 1.0], [1.0, 0.0, 0.0]]
        summary = PRECISION.summarize(rows, truth)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["top1_agreements"], 2)
        self.assertEqual(summary["cosines"][0], 1.0)
        self.assertEqual(summary["minimum_decision"], 2)
        self.assertAlmostEqual(summary["minimum"], 1 / math.sqrt(2), places=15)
        tied = PRECISION.summary([0.5, 0.25, 0.25], 0)
        self.assertEqual(tied["minimum_decision"], 1)
        with self.assertRaises(ValueError):
            PRECISION.summary([], 0)
        with self.assertRaises(ValueError):
            PRECISION.summarize(rows, truth[:2])

    def test_rows_decode_little_endian_and_check_their_length(self) -> None:
        """Rows round-trip through binary32 and a truncated payload is refused."""

        rows = [[1.5, -2.0, 0.25], [3.0, 4.0, -0.5]]
        self.assertEqual(PRECISION.decode_rows(rows_payload(rows), 2, width=3), rows)
        with self.assertRaises(ValueError):
            PRECISION.decode_rows(rows_payload(rows)[:-4], 2, width=3)

    def test_oracle_reader_authenticates_every_file(self) -> None:
        """Token IDs come only from a bundle whose four files match their pinned digests."""

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            vocabulary = PRECISION.VOCABULARY
            record = {"input_ids": [1, 2, 3], "generation": {"token_ids": list(range(PRECISION.DECISIONS))}}
            files = {
                "reference.json": json.dumps(record).encode(),
                "prompt-logits.f32le": bytes(4 * vocabulary),
                "teacher-prefix-logits.f32le": bytes(4 * vocabulary * PRECISION.DECISIONS),
                "cached-prefix-logits.f32le": bytes(4 * vocabulary * PRECISION.DECISIONS),
            }
            for suffix, payload in files.items():
                (root / f"{PRECISION.ORACLE_STEM}{suffix}").write_bytes(payload)
            pinned = {suffix: hashlib.sha256(payload).hexdigest() for suffix, payload in files.items()}
            oracle = PRECISION.read_oracle(root, pinned)
            self.assertEqual(oracle["prompt_ids"], [1, 2, 3])
            self.assertEqual(oracle["generated_ids"], list(range(PRECISION.DECISIONS)))
            self.assertEqual(len(oracle["payloads"]["teacher"]), 4 * vocabulary * PRECISION.DECISIONS)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                PRECISION.read_oracle(root, {**pinned, "reference.json": "0" * 64})
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                PRECISION.read_oracle(root)

    def test_publication_never_replaces_a_bundle(self) -> None:
        """Publishing onto an existing directory fails and leaves both directories."""

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            staging, destination = root / ".staging", root / "bundle"
            staging.mkdir()
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                PRECISION.publish(staging, destination)
            self.assertTrue(staging.is_dir() and destination.is_dir())
            destination.rmdir()
            PRECISION.publish(staging, destination)
            self.assertTrue(destination.is_dir() and not staging.exists())

    def test_offload_arguments_come_together(self) -> None:
        """A resident budget needs an offload directory and vice versa."""

        base = ["--model-dir", "m", "--transformers-dir", "t", "--oracle-dir", "o", "--output", "b"]
        arguments = PRECISION.parse_arguments(base)
        self.assertIsNone(arguments.max_resident_gib)
        offloaded = PRECISION.parse_arguments(base + ["--max-resident-gib", "16", "--offload-dir", "d"])
        self.assertEqual((offloaded.max_resident_gib, offloaded.offload_dir), (16.0, Path("d")))
        for partial in (["--max-resident-gib", "16"], ["--offload-dir", "d"]):
            with self.assertRaises(SystemExit):
                PRECISION.parse_arguments(base + partial)

    def test_evaluation_plan_widens_only_from_bfloat16(self) -> None:
        """F32 is evaluated from BF16-resident weights; F16 and BF16 run as loaded."""

        plan = {label: (load, evaluation) for label, load, evaluation in PRECISION.EVALUATIONS}
        self.assertEqual(plan["float32"], ("bfloat16", "float32"))
        self.assertEqual(plan["bfloat16"], ("bfloat16", "bfloat16"))
        self.assertEqual(plan["float16"], ("float16", "float16"))
        self.assertEqual([kind for kind, _ in PRECISION.KINDS], ["prompt", "teacher", "cached"])


@unittest.skipUnless(HAS_TORCH, "needs PyTorch")
class BlockWideningTests(unittest.TestCase):
    """Check `blocks_in_dtype` on a toy model shaped like the official class."""

    @staticmethod
    def toy(torch):
        """A BF16 model with `model.embeddings`, `model.layers`, and `lm_head`, plus an F32 buffer."""

        class Inner(torch.nn.Module):
            """Embedding and three linear blocks."""

            def __init__(self) -> None:
                """Builds the toy backbone."""

                super().__init__()
                self.embeddings = torch.nn.Embedding(16, 8)
                self.layers = torch.nn.ModuleList(torch.nn.Linear(8, 8) for _ in range(3))

        class Toy(torch.nn.Module):
            """Backbone plus head."""

            def __init__(self) -> None:
                """Builds the toy model."""

                super().__init__()
                self.model = Inner()
                self.lm_head = torch.nn.Linear(8, 16, bias=False)
                self.register_buffer("correction", torch.tensor([0.1, 0.2]))

            def forward(self, tokens):
                """Embeds, runs each block, and projects."""

                hidden = self.model.embeddings(tokens)
                for block in self.model.layers:
                    hidden = torch.tanh(block(hidden))
                return self.lm_head(hidden)

        torch.manual_seed(0)
        model = Toy()
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
        return model

    def test_widened_blocks_equal_a_permanent_cast_and_are_restored(self) -> None:
        """Results match a once-widened model bit for bit; afterwards everything is BF16 again."""

        import torch

        tokens = torch.tensor([[1, 2, 3, 4]])
        streamed, widened = self.toy(torch), self.toy(torch)
        for parameter in widened.parameters():
            parameter.data = parameter.data.to(torch.float32)
        # Simulate Accelerate on one block: an outer forward calling `_old_forward`.
        block = streamed.model.layers[1]
        block._old_forward = block.forward
        block.forward = lambda *arguments, **keywords: block._old_forward(*arguments, **keywords)
        with torch.no_grad(), PRECISION.blocks_in_dtype(torch, streamed, torch.float32):
            self.assertEqual(streamed.model.embeddings.weight.dtype, torch.float32)
            self.assertEqual(streamed.model.layers[0].weight.dtype, torch.bfloat16)
            result = streamed(tokens)
        with torch.no_grad():
            expected = widened(tokens)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual({parameter.dtype for parameter in streamed.parameters()}, {torch.bfloat16})
        self.assertEqual(streamed.correction.dtype, torch.float32)
        self.assertNotIn("forward", vars(streamed.model.layers[0]))
        self.assertIn("forward", vars(block))
        with torch.no_grad():
            self.assertEqual(streamed(tokens).dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
