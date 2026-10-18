"""Validate the October 18 half-precision review record without checkpoints or a GPU.

The record is published with the crate. Its run logs, mutation logs, probe,
benchmark comparison, preconditions, and source-validation logs live in the
repository-only campaign directory
``acceptance/measurements/half-precision-review-20261018/``, so every read of
them goes through :func:`repository_evidence.repository_evidence`; in the
extracted package those tests skip, and in a Git checkout a missing file
fails them. The tests recompute every run's status, source identity, and
binary from its log, every quote, the benchmark comparison from the
envelope round's committed benchmark files (authenticated by that record's
digests) and the approved September 10 summaries, the mutation outcomes, and
the source checks; they tie the probe's Transformers bits to the constants
the Rust test pins, and require the earlier records to keep their bytes.
"""

import functools
import hashlib
import json
from pathlib import Path
import re
import statistics
import unittest

from repository_evidence import repository_evidence


ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = ROOT / "acceptance/measurements"
RECORD = MEASUREMENTS / "half-precision-review-20261018.json"
ENVELOPE = MEASUREMENTS / "half-precision-envelope-20261018.json"
FOLLOWUPS = MEASUREMENTS / "half-precision-followups-20261017.json"
BASELINE = MEASUREMENTS / "small-model-performance-20260910"
#: The Rust test that pins the probe's Transformers bits.
DISPATCH_TESTS = ROOT / "src/attention/dispatch_tests.rs"
#: Records the round must leave untouched.
HISTORICAL = frozenset({
    "half-precision-calibration-20261017.json",
    "mac-validation-20261017.json",
    "half-precision-envelope-20261018.json",
    "half-precision-followups-20261017.json",
})
#: The message each expected failure must show.
EXPECTED_FAILURE = {
    "uncalibrated.": "uncalibrated: official cpu-eager float16 non-finite",
    "refusal.": "backend cannot store checkpoint dtype BF16",
}


def sha256(data):
    """Hexadecimal SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def quoted_entries(value):
    """Every ``{"log": ..., "quotes": [...]}`` mapping anywhere in ``value``."""
    if isinstance(value, dict):
        if isinstance(value.get("log"), str) and isinstance(value.get("quotes"), list):
            yield value
        for child in value.values():
            yield from quoted_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from quoted_entries(child)


class HalfPrecisionReviewRecordTests(unittest.TestCase):
    """Tie every status, quote, figure, and identity of the review record to its committed evidence."""

    @classmethod
    def setUpClass(cls):
        """Read only the published records; campaign files are read on demand."""
        cls.record = json.loads(RECORD.read_text())
        cls.envelope = json.loads(ENVELOPE.read_text())
        cls.campaign = cls.record["campaign_directory"]

    def evidence(self, relative):
        """Bytes of a campaign file named relative to the campaign directory, checked against ``files``."""
        data = repository_evidence(MEASUREMENTS / self.campaign / relative).read_bytes()
        self.assertEqual(sha256(data), self.record["files"][relative], relative)
        return data

    @functools.lru_cache(maxsize=None)
    def text(self, relative):
        """Text of an authenticated campaign file."""
        return self.evidence(relative).decode()

    def envelope_json(self, relative):
        """A file of the envelope campaign, authenticated by the envelope record's digest."""
        path = repository_evidence(MEASUREMENTS / self.envelope["campaign_directory"] / relative)
        data = path.read_bytes()
        self.assertEqual(sha256(data), self.envelope["files"][relative], relative)
        return json.loads(data)

    def test_record_is_evidence_not_a_support_declaration(self):
        """The record promotes nothing, does not pass, and names what it did not run and what awaits the maintainer."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertFalse(record["advertised_supported"])
        self.assertFalse(record["passed"])
        self.assertRegex(record["source_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(record["campaign_directory"], "half-precision-review-20261018")
        self.assertTrue(record["not_run"])
        items = [entry["item"] for entry in record["awaiting_maintainer"]]
        self.assertTrue(any("Re-baselining Qwen3.5-0.8B's cold first token" in item for item in items))
        self.assertTrue(any("Gemma 3 F16 policy" in item for item in items))
        self.assertEqual({issue["id"] for issue in record["review"]},
                         {"benchmark-breach", "nemotron-docs", "mixer-silu-tests", "eager-rounding-points",
                          "eager-scale-users", "gemma-f16-refusal", "pytorch-scale-evidence"})

    def test_files_cover_the_campaign_directory(self):
        """``files`` lists exactly the campaign directory's files, each with its digest."""
        root = repository_evidence(MEASUREMENTS / self.campaign / "gates/logs/runs.jsonl").parents[2]
        present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        self.assertEqual(present, set(self.record["files"]))
        for relative in sorted(present):
            with self.subTest(file=relative):
                self.evidence(relative)

    def test_every_quote_is_in_its_log(self):
        """Every quote anywhere in the record is a line of the log it cites, or part of one."""
        for entry in quoted_entries(self.record):
            lines = self.text(entry["log"]).splitlines()
            for quote in entry["quotes"]:
                with self.subTest(log=entry["log"], quote=quote[:60]):
                    self.assertGreaterEqual(len(quote), 20)
                    self.assertTrue(any(quote in line for line in lines), quote)

    def test_runs_have_their_recorded_status_source_and_binary(self):
        """Each run's log names the code head, a clean checkout, a recorded binary, and the recorded outcome."""
        binaries = {binary["path"]: binary["sha256"] for binary in self.record["binaries"]}
        build = self.text("gates/logs/build.log")
        self.assertIn(f"== source {self.record['source_commit']} dirty=0", build.splitlines())
        self.assertIn(f"== source after build {self.record['source_commit']} dirty=0", build.splitlines())
        jsonl = [json.loads(line) for line in self.text("gates/logs/runs.jsonl").splitlines()]
        self.assertEqual([run["name"] for run in jsonl], [run["name"] for run in self.record["runs"]])
        for run in self.record["runs"]:
            with self.subTest(run=run["name"]):
                text = self.text(run["log"])
                lines = text.splitlines()
                self.assertEqual(lines[0], f"== source {self.record['source_commit']} dirty=0")
                self.assertIn(f"== source after run {self.record['source_commit']} dirty=0", lines)
                path, digest = re.search(r"^== binary (.+) sha256 ([0-9a-f]{64})$", text, re.M).groups()
                self.assertEqual(binaries[path], digest)
                status = int(re.search(r"^== exit=(-?\d+)", text, re.M).group(1))
                self.assertEqual(status, run["exit"])
                result = "pass" if "test result: ok. 1 passed" in text else "fail"
                self.assertEqual(result, run["result"])
                self.assertEqual(result, run["expected"])
                self.assertNotIn("watchdog=", text)
                for prefix, message in EXPECTED_FAILURE.items():
                    if run["name"].startswith(prefix):
                        self.assertEqual(run["expected"], "fail")
                        self.assertIn(message, text)
        runs = self.record["runs"]
        self.assertEqual(self.record["tallies"], {
            "runs": len(runs), "pass": sum(r["result"] == "pass" for r in runs),
            "fail": sum(r["result"] == "fail" for r in runs),
            "as_expected": sum(r["result"] == r["expected"] for r in runs)})
        self.assertEqual(self.record["tallies"]["as_expected"], len(runs))

    def test_every_gemma_gate_ran_and_reproduced_8181605(self):
        """All 20 gemma-3-4b-it gates ran; the 17 with an 8181605 counterpart reproduce it exactly."""
        names = {run["name"] for run in self.record["runs"]}
        tests = ["prompt_logits_match_the_reference", "cached_greedy_matches_the_reference",
                 "batched_prefill_matches_the_reference", "long_context_logits_cross_the_sliding_window",
                 "long_context_cached_decode_wraps_the_rings"]
        for profile in ("f32.flex", "f32.metal", "bf16.flex"):
            for test in tests:
                self.assertIn(f"gemma.{profile}.{test}", names)
        for test in tests:
            self.assertIn(f"uncalibrated.gemma.f16.metal.{test}", names)
        compare = self.text("gates/logs/compare.log").splitlines()
        self.assertIn('== tally {"identical": 17, "different": 0}', compare)

    def test_benchmark_breach_is_recomputed_from_the_committed_benchmarks(self):
        """The cold-first-token figures follow from the envelope round's benchmarks and the approved summaries."""
        manifest = json.loads((BASELINE / "manifest.json").read_text())
        breach = self.text("benchmark/breach.log").splitlines()
        for kind, summary_name, gate_name in (
                ("baseline", "qwen-controlled-current-summary.json", "qwen35-baseline-benchmark.json"),
                ("public", "qwen-public-summary.json", "qwen35-public-causal-greedy-benchmark.json")):
            data = (BASELINE / summary_name).read_bytes()
            self.assertEqual(sha256(data), manifest["records"][summary_name]["sha256"])
            approved = json.loads(data)["benchmark"]["cold_first_token"]["seconds"]
            gate = self.envelope_json(f"gates/bench/{gate_name}")["cold_first_token"]["seconds"]
            samples = {side: [self.envelope_json(f"gates/control/bench/r{n}.{side}.{kind}.json")
                              ["cold_first_token"]["seconds"] for n in (1, 2, 3)] for side in ("base", "head")}
            above = {side: sum(v > 1.10 * approved for v in values) for side, values in samples.items()}
            self.assertGreater(gate, 1.10 * approved, kind)
            self.assertIn(f"   gate run (f9f78b5): {gate:.6f} s ({gate / approved - 1:+.1%}) ABOVE-10%", breach)
            every = samples["base"] + samples["head"]
            ratio = statistics.median(samples["head"]) / statistics.median(samples["base"])
            self.assertIn(f"   controlled rerun, both binaries: {above['base'] + above['head']} of 6 samples above 1.10x "
                          f"(range {min(every):.4f} to {max(every):.4f} s); head/base median ratio {ratio:.3f}", breach)
            self.assertLess(ratio, 1.0, "the head is not slower than the base binary")
            self.assertGreater(statistics.median(samples["base"]), 1.10 * approved, "the base binary breaches too")
            if kind == "baseline":
                self.assertEqual(above, {"base": 3, "head": 3})
            else:
                self.assertEqual(above, {"base": 2, "head": 0})
        decision = json.loads(FOLLOWUPS.read_text())["decisions"][0]
        self.assertEqual(decision["id"], "a")
        self.assertIn(self.record["benchmark_breach"]["decision_source"]["quote"], decision["rationale"])

    def test_mutations_fail_and_baselines_pass(self):
        """Each mutation's test fails and the tree is restored; each unmutated test passes."""
        runs = self.record["mutations"]["runs"]
        self.assertEqual(sum(run["expected"] == "fail" for run in runs), 7)
        for run in runs:
            with self.subTest(log=run["log"]):
                text = self.text(run["log"])
                status = int(re.search(r"^== exit=(-?\d+)", text, re.M).group(1))
                self.assertEqual(status, run["exit"])
                lines = text.splitlines()
                self.assertTrue(lines[0].startswith(f"== source {self.record['source_commit']}"))
                if run["expected"] == "fail":
                    self.assertNotEqual(status, 0)
                    self.assertIn("== restored, git status: clean", lines)
                    self.assertIn("== diff", lines)
                else:
                    self.assertEqual(status, 0)

    def test_probe_bits_are_the_rust_test_constants(self):
        """The Transformers eager bits the probe printed are the constants the Rust test pins."""
        log = self.text("probes/probe_torch_half.log")
        source = DISPATCH_TESTS.read_text()
        for dtype, name in (("bfloat16", "TRANSFORMERS_EAGER_BF16"), ("float16", "TRANSFORMERS_EAGER_F16")):
            bits = re.search(rf"^torch\.{dtype} output bits \[batch, head, query, value\]: (.*)$", log, re.M).group(1)
            body = re.search(rf"const {name}: \[u16; 96\] = \[(.*?)\];", source, re.S).group(1)
            pinned = [token.strip().removeprefix("0x") for token in body.split(",") if token.strip()]
            self.assertEqual(bits.split(), pinned, name)
        self.assertIn(f"sha256 {sha256(self.evidence('probes/probe_torch_half.py'))}", log.splitlines()[0])

    def test_source_validation_passed_at_the_code_head(self):
        """Every source check ran from a clean tree at the code head and exited zero."""
        validation = self.record["source_validation"]
        self.assertEqual(validation["commit"], self.record["source_commit"])
        summary = self.text("validation/summary.log").splitlines()
        self.assertTrue(summary[0].startswith(f"== source {self.record['source_commit']} dirty=0 "))
        names = [check["name"] for check in validation["checks"]]
        self.assertEqual(names, ["fmt", "clippy-nightly", "clippy-1.99.1", "clippy-1.99.1-cpu", "test-default",
                                 "test-features", "test-autodiff-lib", "msrv-1.92.0", "rustdoc-private",
                                 "rustdoc-public", "doc-audit", "python-tests", "release-source"])
        for check in validation["checks"]:
            with self.subTest(check=check["name"]):
                lines = self.text(check["log"]).splitlines()
                self.assertEqual(lines[0], f"== source {self.record['source_commit']} dirty=0")
                self.assertEqual(lines[1], f"== cmd {check['command']}")
                self.assertEqual(lines[-1], f"== exit={check['exit']}")
                self.assertEqual(check["exit"], 0)
        self.assertTrue(validation["passed"])

    def test_historical_records_keep_their_bytes(self):
        """The records this one corrects or extends keep their published bytes."""
        self.assertEqual(set(self.record["historical_records"]), HISTORICAL)
        for name, digest in self.record["historical_records"].items():
            with self.subTest(record=name):
                self.assertEqual(sha256((MEASUREMENTS / name).read_bytes()), digest)


if __name__ == "__main__":
    unittest.main()
