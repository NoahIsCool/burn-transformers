"""Pure release-runner tests: no Cargo commands, network, or GPU required."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import tomllib
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

    def test_successful_export_with_warnings_is_not_qualified(self):
        """LLVM's zero status does not excuse stale, mismatched, or partial data."""

        for diagnostic in (
            "warning: 30 functions have mismatched data",
            "  warning: profile data may be out of date",
            "llvm-cov: warning: malformed profile",
            "/tmp/source.rs:12:3: warning: incomplete coverage",
            "warning[unused]: instrumentation issue",
            "\x1b[33mwarning\x1b[0m: stale profile",
            "error: failed to merge",
            "llvm-profdata: error: corrupt counters",
            "WARNING: partial data",
        ):
            with self.subTest(diagnostic=diagnostic), self.assertRaisesRegex(ValueError, "unqualified"):
                GATES.require_clean_coverage_log(f"running 10 tests\n{diagnostic}\nfinished\n")
        GATES.require_clean_coverage_log(
            "test source::warning::context ... ok\n"
            "test errors::tests::error_is_contextual ... ok\n"
            "Finished test profile\nSummary: 30 functions\n"
        )

    def test_coverage_validation_failure_records_native_success_separately(self):
        """An invalid export must leave a failed manifest and skip later gates."""

        cases = (
            ("warning: 30 functions have mismatched data\n", None, "unqualified"),
            ("finished\n", "{", "Expecting"),
            ("finished\n", None, "coverage.json"),
            ("finished\n", '{"data": []}', "empty coverage"),
        )
        for log, report, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                crate = Path(temporary)

                def run(command, **kwargs):
                    """Emit a synthetic native result without running Cargo or LLVM."""

                    kwargs["stdout"].write(log)
                    if report is not None:
                        (Path(kwargs["stdout"].name).parent / "coverage.json").write_text(report)
                    return subprocess.CompletedProcess(command, 0)

                with (
                    mock.patch.object(GATES, "CRATE", crate),
                    mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "coverage", "--gate", "source"]),
                    mock.patch.object(GATES.subprocess, "run", side_effect=run) as process,
                    mock.patch("builtins.print"),
                ):
                    self.assertEqual(GATES.main(), 1)
                self.assertEqual(process.call_count, 1)
                manifest, = (crate / "target/release-gates").glob("*/results.json")
                result = json.loads(manifest.read_text())
                self.assertFalse(result["passed"])
                self.assertEqual(result["gates"][0]["process_exit_code"], 0)
                self.assertEqual(result["gates"][0]["exit_code"], 1)
                self.assertIn(expected, result["gates"][0]["validation_error"])
                self.assertNotIn("coverage", result["gates"][0])

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

    def test_package_patterns_are_root_anchored(self):
        """Unanchored filename globs also select artifacts through model symlinks."""

        manifest = tomllib.loads((GATES.CRATE / "Cargo.toml").read_text())
        self.assertTrue(all(pattern.startswith("/") for pattern in manifest["package"]["include"]))

    def test_package_listing_checks_the_actual_cargo_inventory(self):
        """Allow model source but reject artifact trees, weights, and bytecode."""

        base = "Cargo.toml\nCargo.lock\nREADME.md\nsrc/lib.rs\n"
        self.assertEqual(GATES.validate_package_listing(base + "src/models/t5.rs\n"), 5)
        self.assertEqual(GATES.validate_package_listing("warning: manifest has no homepage\n  |\n  = note: docs\n" + base), 4)
        for entry in ("models/model/README.md", "target/package.md", "fuzz/README.md", "nested/Cargo.lock", "tests/weights.safetensors", "tests/weights.bpk", "scripts/__pycache__/x.pyc", "/tmp/README.md", "src/../secret.md", "Cargo.lock"):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                GATES.validate_package_listing(base + entry + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            GATES.validate_package_listing("Cargo.toml\n")


if __name__ == "__main__":
    unittest.main()
