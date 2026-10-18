"""Tests for the official-graph half-precision envelope derivation, on synthetic bundles.

Every bundle here is written by `write_bundle` with a two-entry vocabulary, so
each statistic can be checked by hand. No checkpoint, PyTorch, or Transformers
is needed; NumPy is.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "derive_half_precision_envelope.py"
SPEC = importlib.util.spec_from_file_location("derive_half_precision_envelope", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ENVELOPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENVELOPE)

#: Whether NumPy, which the derivation reads logits with, is available.
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
#: Token IDs of the two synthetic prompts; prompt 1 is the longest.
PROMPTS = [[7, 8], [7, 8, 9, 10]]
#: Greedy tokens of the synthetic truth.
GREEDY = [3, 4]
#: Truth logits of each prompt, `[tokens, 2]` flattened.
TRUTH_PROMPTS = [[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]
#: Decode-truth stateless rows `[2 steps, 2]`.
TRUTH_DECODE = [5.0, 6.0, 7.0, 8.0]


def f32le(values) -> bytes:
    """Encodes floats as little-endian binary32."""

    return struct.pack(f"<{len(values)}f", *values)


def write_bundle(root: Path, name: str, dtype: str, device: str = "cpu", attention: str = "eager",
                 prompts=None, decode=None, prompt_decode=None, script: str = "s" * 64,
                 tokens_source: str | None = None, edit=None) -> Path:
    """Writes a schema-1 bundle; `decode` is `(cached, stateless)` rows and `prompt_decode` the longest prompt's rows."""

    directory = root / name
    directory.mkdir()
    prompts = TRUTH_PROMPTS if prompts is None else prompts
    manifest = {
        "schema_version": 1,
        "model_type": "llama",
        "model_class": "LlamaForCausalLM",
        "vocabulary_size": 2,
        "provenance": {"dtype": dtype, "device": device, "attention_implementation": attention,
                       "bf16_resident": False, "num_hidden_layers": None, "config_sha256": "c0",
                       "weights_sha256": {"model.safetensors": "w0"}, "repository": "o/n", "revision": "r0",
                       "transformers": "5.2.0", "torch": "2.10.0", "script_sha256": script},
        "prompts": [],
        "greedy": {"prompt_index": 0, "steps": [{"token": token, "margin": 1.0} for token in GREEDY]},
    }
    for index, (ids, logits) in enumerate(zip(PROMPTS, prompts)):
        payload = f32le(logits)
        (directory / f"logits-{index}.f32le").write_bytes(payload)
        manifest["prompts"].append({"text": f"p{index}", "token_ids": ids, "logits_file": f"logits-{index}.f32le",
                                    "logits_shape": [len(ids), 2], "logits_sha256": hashlib.sha256(payload).hexdigest()})
    if decode is not None:
        cached, stateless = (f32le(rows) for rows in decode)
        (directory / "decode-cached.f32le").write_bytes(cached)
        (directory / "decode-stateless.f32le").write_bytes(stateless)
        manifest["decode_consistency"] = {
            "prompt_index": 0, "first_position": 2, "logits_shape": [2, 2],
            "tokens_source": {"manifest_sha256": tokens_source, "dtype": "float32"},
            "cached_logits_file": "decode-cached.f32le", "cached_logits_sha256": hashlib.sha256(cached).hexdigest(),
            "stateless_logits_file": "decode-stateless.f32le",
            "stateless_logits_sha256": hashlib.sha256(stateless).hexdigest(),
            "steps": [{"position": 2 + step, "fed_token": token, "max_abs_gap": 0.0}
                      for step, token in enumerate(GREEDY)],
            "max_abs_gap": 0.0, "finite": True,
        }
    if prompt_decode is not None:
        payload = f32le(prompt_decode)
        (directory / "decode-prompt.f32le").write_bytes(payload)
        manifest["prompt_decode"] = {"prompt_index": 1, "prefill_tokens": 1, "logits_file": "decode-prompt.f32le",
                                     "logits_shape": [4, 2], "logits_sha256": hashlib.sha256(payload).hexdigest(),
                                     "finite": True}
    if edit is not None:
        edit(manifest)
    (directory / "reference.json").write_text(json.dumps(manifest))
    return directory


def offset(values, amount: float, index: int | None = None):
    """Returns `values` with `amount` added to every entry, or to entry `index` only."""

    return [value + amount if index is None or position == index else value for position, value in enumerate(values)]


class Case:
    """A synthetic truth, decode truth, and four official graphs with chosen deviations."""

    def __init__(self, root: Path, dtype: str = "bfloat16", deviations=(0.5, 0.25, 0.125, 0.0625)):
        """Writes the bundles; graph `g` deviates by `deviations[g]` everywhere, decode rows by twice that."""

        self.root, self.dtype = root, dtype
        self.truth = write_bundle(root, "truth", "float32")
        sha = hashlib.sha256((self.truth / "reference.json").read_bytes()).hexdigest()
        self.truth_sha = sha
        self.decode_truth = write_bundle(root, "decode-truth", "float32", decode=(TRUTH_DECODE, TRUTH_DECODE),
                                         tokens_source=sha)
        self.graphs = []
        for (device, attention), amount in zip(ENVELOPE.GRAPHS, deviations):
            self.graphs.append(self.graph(device, attention, amount))

    def graph(self, device: str, attention: str, amount: float, **overrides) -> Path:
        """Writes one official graph deviating by `amount` (decode rows by `2 amount`)."""

        options = dict(
            prompts=[offset(rows, amount) for rows in TRUTH_PROMPTS],
            decode=(offset(TRUTH_DECODE, 2 * amount), offset(TRUTH_DECODE, -2 * amount, 0)),
            prompt_decode=offset(TRUTH_PROMPTS[1], 3 * amount),
            tokens_source=self.truth_sha,
        )
        options.update(overrides)
        return write_bundle(self.root, f"{device}-{attention}-{len(list(self.root.iterdir()))}", self.dtype,
                            device, attention, **options)

    def spec(self, graphs=None) -> dict:
        """The derivation's case description."""

        return {"name": "synthetic", "truth": str(self.truth), "decode_truth": str(self.decode_truth),
                "graphs": [str(graph) for graph in (self.graphs if graphs is None else graphs)]}


class RoundingTests(unittest.TestCase):
    """The slack rounding works on the shortest decimal and rounds up to two significant digits."""

    def test_round_up_to_two_significant_digits(self) -> None:
        """The documented boundaries: 1.4206 to 1.5, 1.2988 to 1.3, and 1.3 stays 1.3."""

        for value, expected in ((1.4206, 1.5), (1.2988, 1.3), (1.3, 1.3), (1.30000001, 1.4), (1.0, 1.0),
                                (1.269, 1.3), (1.114, 1.2), (2.195, 2.2), (0.0412, 0.042), (12.01, 13.0)):
            with self.subTest(value=value):
                self.assertEqual(ENVELOPE.round_up(value), expected)
        # A ratio computed in binary that prints as 1.3 is still 1.3.
        self.assertEqual(ENVELOPE.round_up(0.13 * 10), 1.3)
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ENVELOPE.round_up(bad)


@unittest.skipUnless(HAS_NUMPY, "needs NumPy")
class EnvelopeTests(unittest.TestCase):
    """Envelopes, leave-one-out ratios, slacks, uncalibrated cases, and every pairing refusal."""

    def setUp(self) -> None:
        """Creates a scratch directory per test."""

        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)

    def tearDown(self) -> None:
        """Removes the scratch directory."""

        self.scratch.cleanup()

    def test_envelope_is_the_largest_official_deviation_and_ratios_are_leave_one_out(self) -> None:
        """E is the maximum over the four graphs; each ratio divides by the other three's maximum."""

        case = Case(self.root)
        result = ENVELOPE.derive([case.spec()], "bfloat16")
        report = result["cases"][0]
        self.assertTrue(report["calibrated"])
        prompt = report["statistics"][0]
        self.assertEqual((prompt["kind"], prompt["prompt"]), ("prompt", 0))
        self.assertEqual(prompt["envelope"], {"max": 0.5, "mean": 0.5})
        # cpu-eager deviates by 0.5 against the others' largest 0.25: ratio 2.
        self.assertEqual(prompt["leave_one_out"]["cpu-eager"], {"max": 2.0, "mean": 2.0})
        self.assertEqual(prompt["leave_one_out"]["cpu-sdpa"], {"max": 0.5, "mean": 0.5})
        decode = report["statistics"][2]
        self.assertEqual(decode["kind"], "decode")
        # Cached rows deviate by 2a everywhere, stateless by 2a at one entry of four: max 1, mean max(1, 0.25).
        self.assertEqual(decode["envelope"], {"max": 1.0, "mean": 1.0})
        self.assertEqual(report["graphs"]["cpu-eager"]["decode_passes"]["stateless"]["mean"], 0.25)
        prompt_decode = report["statistics"][3]
        self.assertEqual((prompt_decode["kind"], prompt_decode["envelope"]), ("prompt_decode", {"max": 1.5, "mean": 1.5}))
        slacks = result["slacks"]
        self.assertEqual((slacks["max"]["slack"], slacks["mean"]["slack"]), (2.0, 2.0))
        self.assertEqual(slacks["max"]["count"], 4 * 4)
        self.assertEqual(set(slacks["max"]["largest_by_kind"]), set(ENVELOPE.KINDS))
        self.assertEqual(result["uncalibrated"], [])
        text = ENVELOPE.render(result)
        self.assertIn("max slack 2.0 from 2.0000 (synthetic prompt 0, cpu-eager)", text)
        self.assertIn("- decode rows: E max 1, mean 1", text)
        self.assertIn("- prompt-decode rows: E max 1.5", text)
        json.dumps(result, allow_nan=False)

    def test_the_maximum_records_its_first_position_and_token(self) -> None:
        """The deviation's maximum names the first entry attaining it, in row-major order."""

        import numpy

        truth = numpy.zeros((3, 2), dtype="<f4")
        actual = numpy.array([[0.0, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype="<f4")
        self.assertEqual(ENVELOPE.deviation(actual, truth), {"max": 1.0, "position": 1, "token": 0, "mean": 2.5 / 6})
        with self.assertRaisesRegex(ENVELOPE.PairingError, "reference logits hold a non-finite"):
            ENVELOPE.deviation(truth, numpy.full((3, 2), numpy.inf, dtype="<f4"))
        with self.assertRaisesRegex(ENVELOPE.PairingError, "shapes"):
            ENVELOPE.deviation(truth[:2], truth)

    def test_a_non_finite_official_graph_leaves_the_case_uncalibrated(self) -> None:
        """One NaN in one graph's prompt makes the case uncalibrated and keeps it out of the slacks."""

        case = Case(self.root, dtype="float16")
        good = ENVELOPE.analyze_case("good", *self.bundles(case), "float16")
        poisoned = list(case.graphs)
        nan = float("nan")
        poisoned[2] = case.graph("mps", "eager", 0.125,
                                 prompts=[TRUTH_PROMPTS[0], [0.0, 1.0, 0.0, 1.0, nan, 1.0, 0.0, 1.0]])
        result = ENVELOPE.derive([case.spec(poisoned), case.spec()], "float16")
        bad = result["cases"][0]
        self.assertFalse(bad["calibrated"])
        self.assertEqual(bad["uncalibrated"], [{"graph": "mps-eager", "kind": "prompt", "prompt": 1,
                                                "first_non_finite": {"position": 2, "token": 0}}])
        self.assertEqual(result["uncalibrated"], ["synthetic"])
        self.assertEqual(result["slacks"]["max"]["count"], 16)
        self.assertEqual(result["slacks"]["max"]["slack"], ENVELOPE.pooled_slacks([good])["max"]["slack"])
        self.assertIn("UNCALIBRATED", ENVELOPE.render(result))
        self.assertIn("official mps-eager float16 non-finite in prompt 1 at position 2, token 0", ENVELOPE.render(result))
        everything_bad = ENVELOPE.derive([case.spec(poisoned)], "float16")
        self.assertIsNone(everything_bad["slacks"])

    def test_a_stopped_greedy_continuation_may_be_shorter(self) -> None:
        """A graph whose continuation stopped on a non-finite step may hold fewer steps; otherwise not."""

        case = Case(self.root)

        def stopped(manifest):
            """Keeps one greedy step and records the non-finite stop."""

            manifest["greedy"]["steps"] = manifest["greedy"]["steps"][:1]
            manifest["greedy"]["stopped"] = "non_finite"

        def short(manifest):
            """Keeps one greedy step without a recorded reason."""

            manifest["greedy"]["steps"] = manifest["greedy"]["steps"][:1]

        graphs = list(case.graphs)
        graphs[0] = case.graph("cpu", "eager", 0.5, edit=stopped)
        ENVELOPE.validate_envelope(*self.bundles(case, graphs), "bfloat16")
        graphs[0] = case.graph("cpu", "eager", 0.5, edit=short)
        with self.assertRaisesRegex(ENVELOPE.PairingError, "greedy length 1 differs"):
            ENVELOPE.validate_envelope(*self.bundles(case, graphs), "bfloat16")

    def bundles(self, case: Case, graphs=None):
        """Reads a case's truth, decode truth, and graphs."""

        graphs = case.graphs if graphs is None else graphs
        return (ENVELOPE.Bundle(case.truth), ENVELOPE.Bundle(case.decode_truth), [ENVELOPE.Bundle(g) for g in graphs])

    def refused(self, case: Case, graphs, message: str, dtype: str = "bfloat16") -> None:
        """Asserts that the envelope of `graphs` is refused with `message`."""

        with self.assertRaisesRegex(ENVELOPE.PairingError, message):
            ENVELOPE.validate_envelope(*self.bundles(case, graphs), dtype)

    def test_every_graph_must_appear_exactly_once_in_the_run_dtype(self) -> None:
        """Missing, duplicated, mislabeled, and wrong-dtype graphs are refused."""

        case = Case(self.root)
        ENVELOPE.validate_envelope(*self.bundles(case), "bfloat16")
        self.refused(case, case.graphs[:3], "exactly 4 official graphs")
        duplicate = [case.graphs[0], case.graphs[0], case.graphs[2], case.graphs[3]]
        self.refused(case, duplicate, "cpu/eager appears 2 times")
        self.refused(case, case.graphs, "a bfloat16 evaluation cannot calibrate a float16 run", dtype="float16")
        self.refused(case, case.graphs, "run dtype must be one of", dtype="float32")
        for device, attention, message in (("cuda", "eager", "cpu/eager appears 0 times"),
                                           ("cpu", "flash", "cpu/eager appears 0 times")):
            with self.subTest(device=device, attention=attention):
                graphs = [case.graph(device, attention, 0.5)] + case.graphs[1:]
                self.refused(case, graphs, message)
        relabeled = [write_bundle(self.root, "relabeled", "float16", "cpu", "eager",
                                  prompts=[offset(rows, 0.5) for rows in TRUTH_PROMPTS],
                                  decode=(TRUTH_DECODE, TRUTH_DECODE), tokens_source=case.truth_sha)]
        self.refused(case, relabeled + case.graphs[1:], "a float16 evaluation cannot calibrate a bfloat16 run")
        resident = [case.graph("cpu", "eager", 0.5, edit=lambda m: m["provenance"].update(bf16_resident=True))]
        self.refused(case, resident + case.graphs[1:], "must not be --bf16-resident")

    def test_graphs_must_share_one_generator_script(self) -> None:
        """A graph written by another generator script is refused."""

        case = Case(self.root)
        other = [case.graph("cpu", "eager", 0.5, script="t" * 64)] + case.graphs[1:]
        self.refused(case, other, "2 generator scripts")

    def test_graphs_must_evaluate_the_truth(self) -> None:
        """Any identity, prompt, or decode-record mismatch with the truth is refused."""

        case = Case(self.root)
        edits = [
            ("weight files", lambda m: m["provenance"]["weights_sha256"].update({"x.safetensors": "w1"})),
            ("config.json SHA-256", lambda m: m["provenance"].update(config_sha256="c1")),
            ("revision", lambda m: m["provenance"].update(revision="r1")),
            ("Transformers version", lambda m: m["provenance"].update(transformers="5.1.0")),
            ("prompt 1 token IDs", lambda m: m["prompts"][1].update(token_ids=[7, 8, 9, 11])),
            ("prompt 0 text", lambda m: m["prompts"][0].update(text="other")),
            ("decode positions", lambda m: m["decode_consistency"]["steps"][1].update(position=9)),
            ("the truth's greedy tokens", lambda m: m["decode_consistency"]["steps"][0].update(fed_token=5)),
            ("start at position", lambda m: m["decode_consistency"].update(first_position=3)),
            ("no decode record", lambda m: m.pop("decode_consistency")),
            ("prompt_decode must decode the longest", lambda m: m["prompt_decode"].update(prompt_index=0)),
            ("prefill max", lambda m: m["prompt_decode"].update(prefill_tokens=2)),
        ]
        for message, edit in edits:
            with self.subTest(message=message):
                graphs = [case.graph("cpu", "eager", 0.5, edit=edit)] + case.graphs[1:]
                self.refused(case, graphs, message)
        source = [case.graph("cpu", "eager", 0.5, tokens_source="0" * 64)] + case.graphs[1:]
        self.refused(case, source, "tokens_source must be the truth's manifest")
        missing = [case.graph("cpu", "eager", 0.5, prompt_decode=None)] + case.graphs[1:]
        self.refused(case, missing, "no prompt_decode record")
        # Without any prompt-decode record the case still calibrates prompts and decode rows.
        none = [case.graph(device, attention, amount, prompt_decode=None)
                for (device, attention), amount in zip(ENVELOPE.GRAPHS, (0.5, 0.25, 0.125, 0.0625))]
        report = ENVELOPE.analyze_case("plain", *self.bundles(case, none), "bfloat16")
        self.assertEqual([statistic["kind"] for statistic in report["statistics"]], ["prompt", "prompt", "decode"])

    def test_the_truth_and_decode_truth_are_checked(self) -> None:
        """A half-precision truth, a decode truth from another graph or with other prompt logits, is refused."""

        case = Case(self.root)
        graphs = [ENVELOPE.Bundle(g) for g in case.graphs]
        truth = ENVELOPE.Bundle(case.truth)
        half = ENVELOPE.Bundle(write_bundle(self.root, "half-truth", "bfloat16"))
        with self.assertRaisesRegex(ENVELOPE.PairingError, "truth must be float32 or float64"):
            ENVELOPE.validate_envelope(half, ENVELOPE.Bundle(case.decode_truth), graphs, "bfloat16")
        for name, options, message in (
            ("decode-bf16", {"dtype": "bfloat16"}, "decode truth must be float32 or float64"),
            ("decode-sdpa", {"attention": "sdpa"}, "decode truth must be CPU eager"),
            ("decode-other-logits", {"prompts": [offset(TRUTH_PROMPTS[0], 1e-3), TRUTH_PROMPTS[1]]},
             "prompt 0 logits differ from the truth's"),
            ("decode-other-source", {"tokens_source": "1" * 64}, "tokens_source must be the truth's manifest"),
        ):
            with self.subTest(name=name):
                arguments = {"dtype": "float32", "decode": (TRUTH_DECODE, TRUTH_DECODE), "tokens_source": case.truth_sha}
                arguments.update(options)
                dtype = arguments.pop("dtype")
                decode_truth = ENVELOPE.Bundle(write_bundle(self.root, name, dtype, **arguments))
                with self.assertRaisesRegex(ENVELOPE.PairingError, message):
                    ENVELOPE.validate_envelope(truth, decode_truth, graphs, "bfloat16")
        stopped = ENVELOPE.Bundle(write_bundle(
            self.root, "stopped-truth", "float32", edit=lambda m: m["greedy"].update(stopped="non_finite")))
        with self.assertRaisesRegex(ENVELOPE.PairingError, "stopped early"):
            ENVELOPE.validate_truth(stopped)

    def test_corrupted_or_misshapen_logits_are_refused(self) -> None:
        """A logits file that no longer matches its digest or size is refused when read."""

        case = Case(self.root)
        (case.graphs[1] / "logits-0.f32le").write_bytes(f32le([9.0, 9.0, 9.0, 9.0]))
        with self.assertRaisesRegex(ENVELOPE.PairingError, "is corrupted"):
            ENVELOPE.derive([case.spec()], "bfloat16")
        (case.graphs[1] / "logits-0.f32le").write_bytes(f32le([9.0]))
        with self.assertRaisesRegex(ENVELOPE.PairingError, r"must hold \[2, 2\] F32 logits"):
            ENVELOPE.derive([case.spec()], "bfloat16")
        bundle = ENVELOPE.Bundle(case.graphs[0])
        with self.assertRaisesRegex(ENVELOPE.PairingError, "not a plain file name"):
            bundle.rows("../logits-0.f32le", "", [2, 2])
        schema = write_bundle(self.root, "schema", "bfloat16", edit=lambda m: m.update(schema_version=2))
        with self.assertRaisesRegex(ENVELOPE.PairingError, "unsupported bundle schema 2"):
            ENVELOPE.Bundle(schema)

    def test_zero_deviation_of_the_other_graphs_is_refused(self) -> None:
        """Three graphs equal to the truth leave the fourth's ratio undefined, which is refused."""

        case = Case(self.root, deviations=(0.5, 0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ENVELOPE.PairingError, "other than cpu-eager have zero max deviation"):
            ENVELOPE.derive([case.spec()], "bfloat16")

    def test_command_line_writes_strict_json_and_prints_the_tables(self) -> None:
        """`main` reads the case list, writes the result, and prints the rendered tables."""

        import contextlib
        import io

        case = Case(self.root)
        cases = self.root / "cases.json"
        cases.write_text(json.dumps([case.spec()]))
        output = self.root / "out.json"
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.assertEqual(ENVELOPE.main(["--dtype", "bfloat16", "--cases", str(cases), "--output", str(output)]), 0)
        result = json.loads(output.read_text())
        self.assertEqual(result["slacks"]["mean"]["slack"], 2.0)
        self.assertIn("# Official bfloat16 graph envelope", printed.getvalue())
        with self.assertRaises(SystemExit):
            ENVELOPE.parse_arguments(["--dtype", "float32", "--cases", "c", "--output", "o"])


if __name__ == "__main__":
    unittest.main()
