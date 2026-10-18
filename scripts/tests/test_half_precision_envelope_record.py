"""Validate the October 18 official half-precision envelope record without checkpoints or a GPU.

The record is published with the crate; its bundle-generation logs,
derivation tables, gate logs, validation logs, probes, and copied harness live
in the repository-only campaign directory
``acceptance/measurements/half-precision-envelope-20261018/``, so every read
of them goes through :func:`repository_evidence.repository_evidence`. In the
extracted package those tests skip; in a Git checkout a missing file still
fails them. The record's slacks are also tied to the Rust module that applies
them and to the derivation script's output, and its generator digest to the
committed generator. Every summary entry that pairs a log with quoted lines,
anywhere in the record, is checked against that log, and the runs that a
harness error made meaningless stay authenticated outside the gate runs.
"""

import functools
import hashlib
import json
from pathlib import Path
import re
import unittest

from repository_evidence import repository_evidence


ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = ROOT / "acceptance/measurements"
RECORD = MEASUREMENTS / "half-precision-envelope-20261018.json"
#: The Rust module whose rule the record measured.
RULE = ROOT / "tests/support/half_precision_calibration.rs"
#: The generator that wrote the record's bundles.
GENERATOR = ROOT / "scripts/generate_causal_lm_reference.py"
#: Historical records the round must leave untouched, by their published digests at the round's base.
HISTORICAL = {
    "half-precision-calibration-20261017.json": "half-precision-calibration",
    "mac-validation-20261017.json": "mac-validation",
}


def sha256(data):
    """Hexadecimal SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def slack_table(source, name):
    """Parses a `(TensorDType::X, value)` slack table named ``name`` from the Rust module."""
    match = re.search(rf"pub\(crate\) const {name}: \[\(TensorDType, f64\); 2\] =\s*\[([^\]]*)\]", source)
    if match is None:
        raise AssertionError(f"{name} is not a two-entry slack table")
    return {dtype: float(value) for dtype, value in re.findall(r"\(TensorDType::(\w+), ([0-9.]+)\)", match.group(1))}


class HalfPrecisionEnvelopeRecordTests(unittest.TestCase):
    """Tie every summarized figure, status, and identity of the record to its committed evidence."""

    @classmethod
    def setUpClass(cls):
        """Read only the published record; campaign files are read on demand."""
        cls.record = json.loads(RECORD.read_text())
        cls.campaign = cls.record["campaign_directory"]

    def evidence(self, relative):
        """Bytes of a repository-only campaign file named relative to the campaign directory."""
        return repository_evidence(MEASUREMENTS / self.campaign / relative).read_bytes()

    @functools.lru_cache(maxsize=None)
    def text(self, relative):
        """Text of a repository-only campaign file."""
        return self.evidence(relative).decode()

    @functools.lru_cache(maxsize=None)
    def json_file(self, relative):
        """A repository-only campaign JSON file."""
        return json.loads(self.evidence(relative))

    def test_record_is_evidence_not_a_support_declaration(self):
        """The record names its sources, promotes nothing, and lists what it did not run."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertFalse(record["advertised_supported"])
        self.assertRegex(record["source_commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(record["cargo_lock_sha256"], r"^[0-9a-f]{64}$")
        commits = [source["commit"] for source in record["sources"]]
        self.assertIn(record["source_commit"], commits)
        for commit in commits:
            self.assertRegex(commit, r"^[0-9a-f]{40}$")
        self.assertTrue(record["not_run"], "unrun checks are listed")
        self.assertTrue(record["awaiting_maintainer"], "user-review decisions are listed")
        self.assertTrue(record["findings"])

    def test_files_match_their_digests_and_cover_the_campaign_directory(self):
        """Every campaign file is authenticated by its recorded SHA-256, and no file is unrecorded."""
        files = self.record["files"]
        for relative, digest in files.items():
            with self.subTest(file=relative):
                self.assertEqual(sha256(self.evidence(relative)), digest)
        directory = repository_evidence(MEASUREMENTS / self.campaign / next(iter(files))).parent
        while directory.name != self.campaign:
            directory = directory.parent
        present = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
        self.assertEqual(present, set(files))

    def test_slacks_are_the_rust_module_constants_and_the_derivation(self):
        """The compiled slacks equal the record's, which equal the derivation script's rounded maxima."""
        source = RULE.read_text()
        compiled = {"maximum": slack_table(source, "MAX_SLACK"), "mean": slack_table(source, "MEAN_SLACK")}
        for dtype, label in (("BF16", "bfloat16"), ("F16", "float16")):
            with self.subTest(dtype=dtype):
                recorded = self.record["slacks"][dtype]
                self.assertEqual(compiled["maximum"][dtype], recorded["maximum"])
                self.assertEqual(compiled["mean"][dtype], recorded["mean"])
                derivation = self.json_file(self.record["derivation"][label]["result"])
                self.assertEqual(derivation["dtype"], label)
                self.assertEqual(derivation["slacks"]["max"]["slack"], recorded["maximum"])
                self.assertEqual(derivation["slacks"]["mean"]["slack"], recorded["mean"])
                top = {stat: derivation["slacks"][stat]["largest"][0] for stat in ("max", "mean")}
                summary = self.record["derivation"][label]
                self.assertEqual(round(top["max"]["ratio"], 4), summary["largest_max_ratio"])
                self.assertEqual(round(top["mean"]["ratio"], 4), summary["largest_mean_ratio"])
                self.assertEqual(sorted(derivation["uncalibrated"]), sorted(summary["uncalibrated"]))
                self.assertIn(f"max slack {recorded['maximum']} from {top['max']['ratio']:.4f}", self.text(summary["tables"]))

    def test_generator_digests_name_committed_bytes(self):
        """The bundles' generator is the committed one, and its default outputs equal round 2's."""
        generator = self.record["generator"]
        self.assertEqual(sha256(GENERATOR.read_bytes()), generator["sha256"])
        self.assertEqual(generator["round2_sha256"], "58ab9f55213720202941fc77de5c06c0a77626e40bc4aabc3c23122bb2e1358b")
        comparison = self.text(generator["default_outputs_identical"]["log"])
        self.assertTrue(comparison.rstrip().endswith("ALL IDENTICAL"))
        self.assertNotIn("DIFFERENT", comparison)

    def test_every_bundle_was_generated_once_and_verified(self):
        """Each new bundle's manifest digest is the one its generation log printed, and verification found no failure."""
        bundles = self.record["bundles"]
        runs = [json.loads(line) for line in self.text(bundles["runs"]).splitlines()]
        by_bundle = {Path(run["bundle"]).name: run for run in runs}
        self.assertEqual(len(by_bundle), len(runs), "no bundle was generated twice")
        for entry in bundles["generated"]:
            with self.subTest(bundle=entry["name"]):
                run = by_bundle[entry["name"]]
                self.assertEqual(run["status"], 0)
                self.assertEqual(run["reference_json_sha256"], entry["reference_json_sha256"])
                self.assertEqual(run["tool_sha256"], self.record["generator"]["sha256"])
                log = self.text(entry["log"])
                self.assertIn(f"== reference.json sha256 {entry['reference_json_sha256']}", log)
                digests = self.json_file(bundles["file_digests"])[entry["name"]]
                self.assertEqual(digests["reference.json"]["sha256"], entry["reference_json_sha256"])
        verification = self.text(bundles["verification"]["log"])
        self.assertTrue(verification.rstrip().endswith("== 0 failures"))
        self.assertEqual(verification.count("\nok   ") + verification.startswith("ok   "), bundles["verification"]["checks"])
        self.assertEqual(sha256(self.evidence(bundles["round2_file_digests"]["file"])), bundles["round2_file_digests"]["sha256"])

    def test_gate_runs_match_their_logs(self):
        """Each run's log names the gate head and binary, and its status and quoted lines are the log's."""
        binaries = {binary["sha256"] for binary in self.record["binaries"]}
        head = self.record["source_commit"]
        for run in self.record["runs"]:
            with self.subTest(run=run["name"]):
                log = self.text(run["log"])
                self.assertEqual(sha256(log.encode()), self.record["files"][run["log"]])
                self.assertTrue(log.startswith(f"== source {head} "), "every gate ran from the gate head's export")
                binary = re.search(r"^== binary \S.* sha256 ([0-9a-f]{64})", log, re.M)
                self.assertIsNotNone(binary)
                self.assertIn(binary.group(1), binaries)
                self.assertIn(f"== {run['kind']} {run['test']}\n", log)
                passed = "test result: ok. 1 passed" in log
                self.assertEqual(passed, run["status"] == "pass", run["status"])
                if run["status"] == "expected_fail":
                    self.assertIn(run["expected_failure"], log)
                for quote in run.get("quotes", []):
                    self.assertIn(quote, log)
        statuses = {run["status"] for run in self.record["runs"]}
        self.assertLessEqual(statuses, {"pass", "fail", "expected_fail"})
        self.assertEqual(
            self.record["passed"],
            all(run["status"] != "fail" for run in self.record["runs"]),
        )

    def test_source_validation_ran_at_its_recorded_commit(self):
        """Every source check ran on a clean tree of a recorded source commit, with the exit its summary and log show."""
        validation = self.record["source_validation"]
        self.assertIn(validation["commit"], [source["commit"] for source in self.record["sources"]])
        summary = self.text(validation["summary"])
        self.assertTrue(summary.startswith(f"== source {validation['commit']} dirty=0"))
        for check in validation["checks"]:
            with self.subTest(check=check["name"]):
                self.assertRegex(summary, rf"(?m)^{re.escape(check['name'])} exit={check['exit']} ")
                log = self.text(check["log"])
                self.assertIn(f"== exit={check['exit']}", log)
        self.assertTrue(all(check["exit"] == 0 for check in validation["checks"]) == validation["passed"])

    def test_superseded_runs_are_kept_but_not_gate_results(self):
        """A harness error's log stays authenticated, apart from the runs, with a reason and no passing test in it."""
        names = {run["name"] for run in self.record["runs"]}
        self.assertTrue(self.record["superseded_runs"])
        for run in self.record["superseded_runs"]:
            with self.subTest(run=run["name"]):
                self.assertNotIn(run["name"], names)
                self.assertTrue(run["reason"])
                self.assertIn(run["log"], self.record["files"])
                self.assertNotIn("test result: ok. 1 passed", self.text(run["log"]))

    def test_every_quoted_line_is_in_its_log(self):
        """Each summary entry outside the gate runs that names a log quotes only lines that log holds."""
        entries = []

        def walk(value, where):
            """Collects every mapping that pairs a log with quoted lines."""
            if isinstance(value, dict):
                if "log" in value and "quotes" in value:
                    entries.append((where, value))
                for key, item in value.items():
                    walk(item, f"{where}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{where}[{index}]")

        for key, value in self.record.items():
            if key != "runs":
                walk(value, key)
        self.assertTrue(entries, "the summary quotes its evidence")
        for where, entry in entries:
            with self.subTest(entry=where):
                self.assertIn(entry["log"], self.record["files"])
                self.assertTrue(entry["quotes"])
                log = self.text(entry["log"])
                for quote in entry["quotes"]:
                    self.assertIn(quote, log)

    def test_highlighted_runs_are_recorded_runs(self):
        """Every highlighted gate line belongs to a recorded run whose log holds it."""
        runs = {run["name"]: run for run in self.record["runs"]}
        self.assertTrue(self.record["highlights"])
        for highlight in self.record["highlights"]:
            with self.subTest(highlight=highlight["run"]):
                run = runs[highlight["run"]]
                self.assertEqual(highlight["status"], run["status"])
                self.assertIn(highlight["quote"], self.text(run["log"]))

    def test_historical_records_are_left_unchanged(self):
        """The round adds a record; the records it supersedes keep the bytes they had at the round's base."""
        for name, digest in self.record["historical_records"].items():
            with self.subTest(record=name):
                self.assertIn(name, HISTORICAL)
                self.assertEqual(sha256((MEASUREMENTS / name).read_bytes()), digest)


if __name__ == "__main__":
    unittest.main()
