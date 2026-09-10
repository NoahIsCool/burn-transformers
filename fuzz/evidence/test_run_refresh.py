"""Deterministic tests of campaign accounting and native-result preservation."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import os

SPEC = importlib.util.spec_from_file_location("run_refresh", Path(__file__).with_name("run_refresh.py"))
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class CampaignRecords(unittest.TestCase):
    """Reject deceptively successful fuzz summaries and preserve native failure evidence."""

    def test_requires_one_exact_terminal_count(self):
        """A zero native exit cannot substitute for all 10,000 executed units."""
        record = {"native_returncode": 0, "timed_out": False}
        empty = {"files": {}}
        valid = "stat::number_of_executed_units: 10000\nstat::peak_rss_mb: 128\n"
        self.assertTrue(RUNNER.campaign_result(record, valid, empty)["passed"])
        for invalid in ["", valid + valid, valid.replace("10000", "9999"),
                        valid.replace("10000", "10001"), "stat::number_of_executed_units: NaN\n"]:
            self.assertFalse(RUNNER.campaign_result(record, invalid, empty)["passed"])

    def test_failure_artifacts_warnings_and_native_errors_are_not_waived(self):
        """Expected execution counts do not erase sanitizer, symbolizer or timeout failures."""
        text = "stat::number_of_executed_units: 10000\n"
        for record in [{"native_returncode": 77, "timed_out": False},
                       {"native_returncode": -15, "timed_out": True}]:
            self.assertFalse(RUNNER.campaign_result(record, text, {"files": {}})["passed"])
        record = {"native_returncode": 0, "timed_out": False}
        for warning in ["WARNING: symbolizer failed", "ERROR: AddressSanitizer", "SUMMARY: timeout"]:
            self.assertFalse(RUNNER.campaign_result(record, text + warning, {"files": {}})["passed"])
        self.assertFalse(RUNNER.campaign_result(record, text, {"files": {"crash-1": {}}})["passed"])

    def test_native_exit_is_retained_and_records_are_create_only(self):
        """Nonzero child exits remain nonzero; an existing record is never overwritten."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = RUNNER.execute("exit", [sys.executable, "-c", "raise SystemExit(7)"],
                                    root, os.environ.copy(), root, 10)
            self.assertEqual(actual["native_returncode"], 7)
            self.assertFalse(actual["timed_out"])
            self.assertEqual(actual["log"], RUNNER.file_identity(root / "exit.log"))
            with self.assertRaises(FileExistsError):
                RUNNER.write_json(root / "exit.command.json", {})

    def test_wall_timeout_preserves_native_signal(self):
        """A bounded supervisor terminates a sleeping child and marks the result failed."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = RUNNER.execute("timeout", [sys.executable, "-c", "import time; time.sleep(10)"],
                                    root, os.environ.copy(), root, 0.1)
            self.assertTrue(actual["timed_out"])
            self.assertLess(actual["native_returncode"], 0)
            self.assertLess(actual["elapsed_seconds"], 5)

    def test_inventory_preserves_content_identity_and_rejects_links(self):
        """Corpus metadata is content-bound, while unreviewed symlinks fail closed."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(RUNNER.inventory(root / "absent")["present"])
            (root / "input").write_bytes(b"abc")
            actual = RUNNER.inventory(root)
            self.assertEqual(actual["total_bytes"], 3)
            self.assertEqual(actual["files"]["input"]["sha256"],
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            (root / "link").symlink_to(root / "input")
            with self.assertRaises(ValueError):
                RUNNER.inventory(root)

    def test_committed_refresh_is_complete_and_consistent(self):
        """The compact record must retain five complete native campaigns and matched identities."""
        record = json.loads(Path(__file__).with_name("refresh-80b51e8.json").read_text())
        self.assertEqual(record["status"], "passed")
        self.assertTrue(record["source_unchanged"])
        self.assertEqual(record["total_executed_units"], 50_000)
        source = record["source_inventory"]
        self.assertEqual(source["before_canonical_json_sha256"], source["after_canonical_json_sha256"])
        self.assertGreater(source["file_count"], 0)
        self.assertEqual(set(record["campaigns"]), {name for name, _ in RUNNER.TARGETS})
        self.assertFalse(record["build_std"])
        self.assertFalse(record["dependency_closure_verified"])
        self.assertFalse(record["historical_corpora_imported"])
        for name, seed in RUNNER.TARGETS:
            campaign = record["campaigns"][name]
            self.assertTrue(campaign["passed"])
            self.assertEqual(campaign["executed_units"], 10_000)
            self.assertEqual(campaign["seed"], seed)
            self.assertEqual(campaign["failures"], [])
            self.assertEqual(campaign["diagnostics"], [])
            self.assertEqual(campaign["initial_corpus"]["file_count"], 0)
            self.assertEqual(campaign["artifacts"]["file_count"], 0)
            self.assertEqual(campaign["executable_before"], campaign["executable_after"])
            command = campaign["command"]
            self.assertEqual(command["native_returncode"], 0)
            self.assertFalse(command["timed_out"])
            self.assertEqual(command["wall_timeout_seconds"], 900)
            self.assertEqual(Path(command["command"][0]).name, name)
            for flag in ["-runs=10000", "-max_len=4096", "-timeout=10", "-rss_limit_mb=2048",
                         f"-seed={seed}", "-print_final_stats=1"]:
                self.assertIn(flag, command["command"])
            self.assertEqual(campaign["final_statistics"]["stat::number_of_executed_units"], "10000")

    def test_retained_raw_report_authenticates_compact_record(self):
        """When external evidence is retained locally, verify its digest and complete counters."""
        compact = json.loads(Path(__file__).with_name("refresh-80b51e8.json").read_text())
        path = Path(compact["raw_report"]["path"])
        if not path.is_file():
            self.skipTest("bulk raw evidence is intentionally not required in a repository checkout")
        self.assertEqual(RUNNER.file_identity(path),
                         {key: compact["raw_report"][key] for key in ("bytes", "sha256")})
        raw = json.loads(path.read_text())
        canonical = lambda value: hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        for phase in ["before", "after"]:
            self.assertEqual(canonical(raw[f"source_{phase}"]),
                             compact["source_inventory"][f"{phase}_canonical_json_sha256"])
        for name, _ in RUNNER.TARGETS:
            actual = raw["campaigns"][name]
            expected = compact["campaigns"][name]
            self.assertEqual(actual["command"], expected["command"])
            self.assertEqual(actual["executed_units"], expected["executed_units"])
            for kind in ["initial_corpus", "final_corpus", "artifacts"]:
                self.assertEqual(canonical(actual[kind]["files"]),
                                 expected[kind]["files_canonical_json_sha256"])
                self.assertEqual(len(actual[kind]["files"]), expected[kind]["file_count"])


if __name__ == "__main__":
    unittest.main()
