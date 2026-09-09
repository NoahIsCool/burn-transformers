"""Pure release-runner tests: no Cargo commands, network, or GPU required."""

import importlib.util
from pathlib import Path
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "check_release.py"
SPEC = importlib.util.spec_from_file_location("check_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATES)


def file_entry(path: str, count: int, covered: int) -> dict:
    """Build only the documented LLVM JSON fields consumed by the runner."""

    return {"filename": str(GATES.CRATE / path), "summary": {"lines": {"count": count, "covered": covered}}}


class ReleaseGateTests(unittest.TestCase):
    """Pin denominator policy, exact threshold comparisons, and safe command plans."""

    def test_exact_thresholds_and_exclusions(self):
        """Keep experimental source but exclude integration harnesses/dependencies."""

        report = {"data": [{"files": [
            file_entry("src/cache/mod.rs", 100, 95),
            file_entry("src/models/muse_glimmer.rs", 100, 85),
            file_entry("tests/cache.rs", 1000, 1000),
            file_entry("target/dependency.rs", 1000, 1000),
        ]}]}
        result = GATES.coverage_summary(report, GATES.CRATE)
        self.assertTrue(result["passed"])
        self.assertEqual(result["crate"]["lines"], 200)
        self.assertEqual(result["crate"]["percent"], 90)
        self.assertEqual(result["shared"]["percent"], 95)
        self.assertFalse(result["branch_coverage_measured"])

    def test_each_threshold_can_fail_independently(self):
        """A passing crate aggregate does not waive the shared-module gate."""

        for shared, model, expected in [(94, 100, (True, False)), (95, 80, (False, True))]:
            report = {"data": [{"files": [file_entry("src/cache.rs", 100, shared), file_entry("src/models/t5.rs", 100, model)]}]}
            result = GATES.coverage_summary(report, GATES.CRATE)
            self.assertEqual((result["crate"]["passed"], result["shared"]["passed"]), expected)
            self.assertFalse(result["passed"])

    def test_empty_and_malformed_reports_fail_closed(self):
        """Do not treat zero lines or invalid counts as perfect coverage."""

        for entries in [[], [file_entry("src/cache.rs", 0, 0)], [file_entry("src/cache.rs", 3, 4)], [file_entry("src/cache.rs", 3, -1)], [file_entry("src/cache.rs", True, True)], [file_entry("src/models/t5.rs", 3, 3)]]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                GATES.coverage_summary({"data": [{"files": entries}]}, GATES.CRATE)

    def test_duplicate_files_cannot_inflate_denominator(self):
        """Require the exporter to merge profiles before reporting counters."""

        entry = file_entry("src/cache.rs", 100, 100)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            GATES.coverage_summary({"data": [{"files": [entry]}, {"files": [entry]}]}, GATES.CRATE)

    def test_all_command_groups_are_scoped(self):
        """Build every group without shell interpolation or mutable lockfiles."""

        for gate in GATES.GATES:
            with self.subTest(gate=gate):
                commands = GATES.command_plan(gate, Path("/tmp/output with spaces"), "stable", "1.92.0", True)
                self.assertTrue(commands)
                self.assertTrue(all(isinstance(command, list) for command in commands))
                for command in commands:
                    self.assertNotIn("--allow-dirty", command)
                    self.assertNotIn("publish", command)
                    self.assertNotIn("--ignored", command)
                if gate in ("source", "msrv", "coverage", "accelerator-compile", "package"):
                    for command in commands:
                        if command[0] == "cargo" and command[2] != "fmt":
                            self.assertIn("--locked", command)
                            self.assertIn("--offline", command)

    def test_offline_advisory_limits_are_explicit(self):
        """Only two reviewed maintenance IDs are waived; stale DBs still fail."""

        command = GATES.command_plan("dependencies", Path("/tmp/output"), "stable", "1.92.0", True)[0]
        self.assertIn("--no-fetch", command)
        self.assertIn("--no-yanked", command)
        self.assertNotIn("--stale", command)
        self.assertEqual(command.count("--ignore"), 2)
        self.assertIn("--deny", command)

    def test_msrv_alias_is_checked_exactly(self):
        """A rolling stable alias must never silently substitute a newer MSRV."""

        with mock.patch.object(GATES.subprocess, "check_output", return_value="rustc 1.92.0 (test)\n"):
            GATES.require_msrv("stable")
        for version in ("rustc 1.93.0", "rustc 1.92.0-nightly", ""):
            with mock.patch.object(GATES.subprocess, "check_output", return_value=version), self.assertRaises(ValueError):
                GATES.require_msrv("stable")

    def test_unknown_group_is_rejected(self):
        """Reject programming errors instead of silently running no gates."""

        with self.assertRaisesRegex(ValueError, "unknown gate"):
            GATES.command_plan("typo", Path("/tmp/output"), "stable", "1.92.0", True)


if __name__ == "__main__":
    unittest.main()
