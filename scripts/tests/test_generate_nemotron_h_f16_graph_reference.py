"""Pure tests for the Nemotron-H official F16 graph tool (no checkpoint; PyTorch only where marked)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "generate_nemotron_h_f16_graph_reference.py"
SPEC = importlib.util.spec_from_file_location("generate_nemotron_h_f16_graph_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)
PRECISION = GRAPH.PRECISION

#: Whether the PyTorch-backed tests can run.
HAS_TORCH = importlib.util.find_spec("torch") is not None


def write_precision_bundle(root: Path, vocabulary: int, decisions: int) -> tuple[Path, str]:
    """Writes a synthetic precision bundle with zeroed rows; returns its directory and manifest digest."""

    bundle = root / GRAPH.PRECISION_BUNDLE
    bundle.mkdir()
    files = {}
    for label in ("float32", "float16", "bfloat16"):
        for kind, count in (("prompt", 1), ("teacher", decisions), ("cached", decisions)):
            name = f"{label}-{kind}-logits.f32le"
            payload = bytes(4 * vocabulary * count)
            (bundle / name).write_bytes(payload)
            files[name] = hashlib.sha256(payload).hexdigest()
    manifest = {"prompt_ids": [1, 2, 3], "generated_ids": list(range(decisions)), "files_sha256": files}
    data = json.dumps(manifest).encode()
    (bundle / "reference.json").write_bytes(data)
    return bundle, hashlib.sha256(data).hexdigest()


class GraphToolTests(unittest.TestCase):
    """Check the authentication, graph selection, and bookkeeping of the graph tool."""

    def test_controls_are_installed_at_import(self) -> None:
        """Importing the tool keeps every MPS operation on MPS and forbids Hub access."""

        self.assertEqual(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_reused_precision_tool_is_the_pinned_script(self) -> None:
        """The repository's precision tool is the exact script that produced the precision bundle."""

        self.assertEqual(GRAPH.verify_precision_tool(), GRAPH.PRECISION_SCRIPT_SHA256)
        with tempfile.TemporaryDirectory() as root:
            copy = Path(root) / "copy.py"
            copy.write_bytes(GRAPH.PRECISION_SCRIPT.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                GRAPH.verify_precision_tool(copy)

    def test_precision_bundle_is_authenticated_file_by_file(self) -> None:
        """The manifest needs its pinned digest, each row file its manifest digest, length, and name."""

        decisions = PRECISION.DECISIONS
        with tempfile.TemporaryDirectory() as root:
            bundle, digest = write_precision_bundle(Path(root), PRECISION.VOCABULARY, decisions)
            read = GRAPH.read_precision_bundle(bundle, digest)
            self.assertEqual(read["manifest_sha256"], digest)
            self.assertEqual(set(read["payloads"]), {"float32", "float16"})
            self.assertEqual(len(read["payloads"]["float32"]["cached"]), 4 * PRECISION.VOCABULARY * decisions)
            with self.assertRaisesRegex(ValueError, "manifest has SHA-256"):
                GRAPH.read_precision_bundle(bundle)
            renamed = Path(root) / "renamed.bundle"
            shutil.copytree(bundle, renamed)
            with self.assertRaisesRegex(ValueError, "must be named"):
                GRAPH.read_precision_bundle(renamed, digest)
            (bundle / "float32-teacher-logits.f32le").write_bytes(bytes(4))
            with self.assertRaisesRegex(ValueError, "not its manifest's"):
                GRAPH.read_precision_bundle(bundle, digest)

    def test_precision_rows_must_have_full_vocabulary_geometry(self) -> None:
        """A row file with the right digest but the wrong length is refused."""

        with tempfile.TemporaryDirectory() as root:
            bundle, digest = write_precision_bundle(Path(root), 2, PRECISION.DECISIONS)
            with self.assertRaisesRegex(ValueError, "full-vocabulary rows"):
                GRAPH.read_precision_bundle(bundle, digest)

    def test_inputs_must_agree_with_the_oracle(self) -> None:
        """The precision bundle must describe the oracle's prompt and tokens."""

        oracle = {"prompt_ids": [1, 2], "generated_ids": [3, 4]}
        GRAPH.check_inputs_agree(oracle, {"manifest": {"prompt_ids": [1, 2], "generated_ids": [3, 4]}})
        for changed in ({"prompt_ids": [1, 9], "generated_ids": [3, 4]}, {"prompt_ids": [1, 2], "generated_ids": [3]}):
            with self.assertRaisesRegex(ValueError, "different inputs"):
                GRAPH.check_inputs_agree(oracle, {"manifest": changed})

    def test_graphs_are_the_envelope_rounds_four_with_their_devices_experts(self) -> None:
        """Only the four graphs load; the CPU keeps the default experts request, MPS the oracle's eager."""

        self.assertEqual(set(GRAPH.GRAPHS), {(d, a) for d in ("cpu", "mps") for a in ("eager", "sdpa")})
        self.assertEqual(GRAPH.load_options("cpu", "sdpa"), {"attn_implementation": "sdpa"})
        self.assertEqual(GRAPH.load_options("mps", "sdpa"),
                         {"attn_implementation": "sdpa", "experts_implementation": "eager"})
        self.assertEqual(GRAPH.resolved_experts("cpu"), "grouped_mm")
        for device, attention in (("cuda", "sdpa"), ("cpu", "flash_attention_2"), ("mps", "flex_attention")):
            with self.assertRaisesRegex(ValueError, "unsupported graph"):
                GRAPH.load_options(device, attention)

    def test_loaded_models_must_resolve_the_requested_graph(self) -> None:
        """A model that fell back to another implementation, lost its F32 buffers, or left MPS is refused."""

        fake_torch = types.SimpleNamespace(float32="f32", float16="f16")

        def model(attn: str, experts: str, buffer_dtype: str = "f32", device: str = "mps", buffers: int = 23):
            """A stand-in exposing the attributes `check_loaded` reads."""

            tensor = types.SimpleNamespace(dtype=buffer_dtype, device=types.SimpleNamespace(type=device))
            return types.SimpleNamespace(
                config=types.SimpleNamespace(_attn_implementation=attn, _experts_implementation=experts),
                named_buffers=lambda: [(f"layers.{i}.e_score_correction_bias", tensor) for i in range(buffers)],
                parameters=lambda: [tensor],
                buffers=lambda: [tensor],
            )

        self.assertEqual(GRAPH.check_loaded(fake_torch, model("sdpa", "eager"), "mps", "sdpa"),
                         {"attention_implementation": "sdpa", "experts_implementation": "eager"})
        GRAPH.check_loaded(fake_torch, model("sdpa", "grouped_mm", device="meta"), "cpu", "sdpa")
        for candidate, device in ((model("eager", "eager"), "mps"), (model("sdpa", "grouped_mm"), "mps"),
                                  (model("sdpa", "eager", buffer_dtype="f16"), "mps"),
                                  (model("sdpa", "eager", buffers=22), "mps"),
                                  (model("sdpa", "eager", device="cpu"), "mps")):
            with self.assertRaises(ValueError):
                GRAPH.check_loaded(fake_torch, candidate, device, "sdpa")
        tiny = model("sdpa", "eager")
        self.assertEqual(GRAPH.check_tiny_loaded(tiny, "mps", "sdpa")["attention_implementation"], "sdpa")
        with self.assertRaises(ValueError):
            GRAPH.check_tiny_loaded(tiny, "cpu", "sdpa")

    def test_arguments_keep_offload_on_the_cpu(self) -> None:
        """A budget needs an offload directory, and MPS refuses one."""

        base = ["--model-dir", "m", "--transformers-dir", "t", "--oracle-dir", "o", "--precision-dir", "p",
                "--output", "b", "--attn-implementation", "sdpa"]
        cpu = GRAPH.parse_arguments(base + ["--device", "cpu", "--max-resident-gib", "16", "--offload-dir", "d"])
        self.assertEqual((cpu.device, cpu.max_resident_gib, cpu.offload_dir), ("cpu", 16.0, Path("d")))
        self.assertIsNone(GRAPH.parse_arguments(base + ["--device", "mps"]).max_resident_gib)
        for extra in (["--device", "cpu", "--max-resident-gib", "16"], ["--device", "cpu", "--offload-dir", "d"],
                      ["--device", "mps", "--max-resident-gib", "16", "--offload-dir", "d"],
                      ["--device", "cuda"]):
            with self.assertRaises(SystemExit):
                GRAPH.parse_arguments(base + extra)

    def test_system_volume_guard_refuses_a_shortfall(self) -> None:
        """The free-space sample passes above its minimum and raises below it."""

        sample = GRAPH.system_free(1, "probe", Path(tempfile.gettempdir()))
        self.assertGreaterEqual(sample["free_bytes"], 1)
        with self.assertRaisesRegex(RuntimeError, "probe"):
            GRAPH.system_free(1 << 62, "probe", Path(tempfile.gettempdir()))

    def test_timed_phases_carry_host_load_samples(self) -> None:
        """Every timed phase records its wall time and a load and top-process sample before and after."""

        samples: list = []
        self.assertEqual(GRAPH.timed(samples, "phase", lambda: 7), 7)
        (entry,) = samples
        self.assertEqual(entry["phase"], "phase")
        self.assertGreaterEqual(entry["seconds"], 0)
        for side in ("before", "after"):
            self.assertEqual(len(entry[side]["load_average"]), 3)
            self.assertTrue(entry[side]["top_cpu"])
            self.assertIn("utc", entry[side])

    def test_publication_never_replaces_a_bundle(self) -> None:
        """Publishing onto an existing directory fails and leaves both directories."""

        with tempfile.TemporaryDirectory() as root:
            staging, destination = Path(root) / ".staging", Path(root) / "bundle"
            staging.mkdir()
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                GRAPH.publish(staging, destination)
            destination.rmdir()
            GRAPH.publish(staging, destination)
            self.assertTrue(destination.is_dir() and not staging.exists())


@unittest.skipUnless(HAS_TORCH, "needs PyTorch")
class DeviceAwareCallTests(unittest.TestCase):
    """`evaluate_on` must issue the precision tool's calls exactly."""

    @staticmethod
    def toy(torch):
        """A tiny causal model with a cache whose logits depend on every earlier token."""

        class Output:
            """The fields the calls read."""

            def __init__(self, logits, past_key_values):
                """Stores the logits and the cache."""

                self.logits, self.past_key_values = logits, past_key_values

        class Toy(torch.nn.Module):
            """Prefix sums of embeddings projected to a vocabulary of 11."""

            def __init__(self) -> None:
                """Builds the toy weights."""

                super().__init__()
                torch.manual_seed(1)
                self.embed = torch.nn.Embedding(128, 5)
                self.head = torch.nn.Linear(5, 11)
                self.calls = []

            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep, return_dict, past_key_values=None):
                """Records the call and returns `logits_to_keep` rows of prefix-sum logits."""

                self.calls.append((input_ids.tolist(), attention_mask.shape[1], use_cache, logits_to_keep,
                                   past_key_values is not None, return_dict))
                hidden = self.embed(input_ids).cumsum(dim=1)
                if past_key_values is not None:
                    hidden = hidden + past_key_values
                logits = self.head(hidden)
                kept = logits if logits_to_keep == 0 else logits[:, -logits_to_keep:]
                return Output(kept, hidden[:, -1:] if use_cache else None)

        return Toy()

    def test_device_aware_calls_equal_the_precision_tools_calls(self) -> None:
        """On the CPU the rows and the call sequence equal the precision tool's ``evaluate`` bit for bit."""

        import torch

        prompt, generated = [3, 1, 4, 1, 5], [9, 2, 6, 5]
        left, right = self.toy(torch), self.toy(torch)
        expected = PRECISION.evaluate(torch, left, prompt, generated)
        actual = GRAPH.evaluate_on(torch, right, prompt, generated, "cpu")
        self.assertEqual(right.calls, left.calls)
        for kind in ("prompt", "teacher", "cached"):
            self.assertTrue(torch.equal(actual[kind], expected[kind]), kind)
        self.assertEqual(list(actual["teacher"].shape), [len(generated), 11])
        self.assertEqual(list(actual["cached"].shape), [len(generated), 11])

    def test_authenticated_payloads_decode_to_row_tensors(self) -> None:
        """Payload bytes decode to `[rows, vocabulary]` F32 tensors in row-major order."""

        import torch

        vocabulary, decisions = PRECISION.VOCABULARY, PRECISION.DECISIONS
        payloads = {kind: bytes(4 * vocabulary * count) for kind, count in PRECISION.KINDS}
        rows = GRAPH.rows_from(torch, payloads)
        self.assertEqual(list(rows["prompt"].shape), [1, vocabulary])
        self.assertEqual(list(rows["cached"].shape), [decisions, vocabulary])
        self.assertEqual(rows["teacher"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
