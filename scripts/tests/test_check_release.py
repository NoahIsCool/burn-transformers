"""Pure release-runner tests: no Cargo commands, network, or GPU required."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import tomllib
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "check_release.py"
SPEC = importlib.util.spec_from_file_location("check_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATES = importlib.util.module_from_spec(SPEC)
with mock.patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    SPEC.loader.exec_module(GATES)


def file_entry(path: str, count: int, covered: int) -> dict:
    """Build only the documented LLVM JSON fields consumed by the runner."""

    return {"filename": str(GATES.CRATE / path), "summary": {"lines": {"count": count, "covered": covered}}}


def minimal_crate(crate: Path) -> None:
    """Provide actual fingerprintable inputs without invoking a compiler."""

    (crate / "src").mkdir()
    (crate / "src/lib.rs").write_text("pub fn value() -> u32 { 7 }\n")
    (crate / "Cargo.toml").write_text("[package]\nname = 'fixture'\n")
    (crate / "Cargo.lock").write_text("version = 4\n")


VERUS_VERSION_OUTPUT = (
    "Verus\n  Version: 0.2026.10.04.426d8b0\n  Profile: release\n"
    "  Platform: macos_aarch64\n  Toolchain: 1.98.1-aarch64-apple-darwin\n"
)


def verus_crate(crate: Path, counts: dict[str, int]) -> None:
    """Write one harness per entry, each including its own source, and their README table."""

    (crate / "src").mkdir()
    (crate / "verus").mkdir()
    rows = []
    for name in sorted(counts):
        stem = name.removesuffix(".rs")
        (crate / f"src/{stem}.rs").write_text(f"pub fn {stem}() -> u8 {{ 1 }}\n")
        (crate / "verus" / name).write_text(f'//! Harness.\n#[path = "../src/{stem}.rs"]\nmod {stem};\n')
        rows.append(f"| `{name}` | `src/{stem}.rs` | {counts[name]:,} | Proofs. |")
    total = sum(counts.values())
    (crate / "verus/README.md").write_text(
        "# Proofs\n\n## Harnesses\n\nThe counts: **" + f"{total:,}" + " verified, 0 errors**.\n\n"
        "| Harness | Verified sources | Verified | What |\n| --- | --- | ---: | --- |\n"
        + "\n".join(rows) + "\n\n## Rules for proof code\n\n| Rejected | Why |\n| --- | --- |\n"
        "| `admit(` | Assumes. |\n"
    )


def fake_verus(counts: dict[str, int], mutate=None):
    """Return a subprocess.run stand-in that prints Verus output for the requested harness."""

    def run(command, **kwargs):
        """Write the version banner or one harness's result line into the gate's log."""

        if command[1:] == ["--version"]:
            kwargs["stdout"].write(VERUS_VERSION_OUTPUT)
        else:
            if mutate is not None:
                mutate(Path(command[-1]))
            kwargs["stdout"].write(f"verification results:: {counts[Path(command[-1]).name]} verified, 0 errors\n")
        return subprocess.CompletedProcess(command, 0)

    return run


class ReleaseGateTests(unittest.TestCase):
    """Pin denominator policy, exact threshold comparisons, and safe command plans."""

    def setUp(self):
        """Keep pure runner tests independent of the host's current free space."""

        patcher = mock.patch.object(GATES.shutil, "disk_usage", return_value=SimpleNamespace(
            free=GATES.MINIMUM_BUILD_FREE_BYTES + 1,
        ))
        self.disk_usage = patcher.start()
        self.addCleanup(patcher.stop)

    def test_compile_detection_includes_coverage_documentation_and_package_verification(self):
        """Every compile-capable operation gets a preflight, even with a warm cache."""

        for operation in ("build", "check", "clippy", "doc", "test", "run", "llvm-cov", "package"):
            for toolchain in ([], ["+stable"]):
                with self.subTest(operation=operation, toolchain=toolchain):
                    self.assertTrue(GATES.command_compiles(["cargo", *toolchain, operation]))
        for command in ([], ["cargo"], ["cargo", "+stable"], ["rustc", "--version"],
                        ["cargo", "+stable", "fmt", "--check"], ["cargo", "metadata"],
                        ["cargo", "audit"], ["cargo", "deny", "check"],
                        ["cargo", "+stable", "package", "--list"]):
            with self.subTest(command=command):
                self.assertFalse(GATES.command_compiles(command))

    def test_storage_preflight_uses_exact_gib_boundary_and_retains_override(self):
        """Exactly 20 GiB passes; a target override never changes the measured path."""

        threshold = 20 * 1024**3
        environment = {"CARGO_TARGET_DIR": "/external/explicit-target", "SECRET": "not recorded"}
        for free, passed in ((threshold - 1, False), (threshold, True), (threshold + 1, True)):
            with self.subTest(free=free):
                self.disk_usage.return_value = SimpleNamespace(free=free)
                observed = GATES.observe_build_storage(GATES.CRATE, environment)
                self.assertEqual(observed["minimum_free_bytes"], threshold)
                self.assertEqual(observed["free_bytes"], free)
                self.assertEqual(observed["passed"], passed)
                self.assertEqual(observed["checked_path"], str(GATES.CRATE.resolve()))
                self.assertTrue(observed["checked_utc"])
                self.assertEqual(observed["target_environment"], {"CARGO_TARGET_DIR": environment["CARGO_TARGET_DIR"]})
                self.disk_usage.assert_called_with(GATES.CRATE)
        self.assertEqual(environment, {"CARGO_TARGET_DIR": "/external/explicit-target", "SECRET": "not recorded"})

    def test_low_storage_stops_before_coverage_setup_or_native_execution(self):
        """A failed storage observation cannot acquire native or compiler-binding claims."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            minimal_crate(crate)
            self.disk_usage.return_value = SimpleNamespace(free=GATES.MINIMUM_BUILD_FREE_BYTES - 1)
            with (
                mock.patch.object(GATES, "CRATE", crate),
                mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "coverage", "--gate", "source", "--coverage-compiler-evidence"]),
                mock.patch.object(GATES, "prepare_capture") as prepare,
                mock.patch.object(GATES, "capture_coverage_context") as capture,
                mock.patch.object(GATES.subprocess, "run") as process,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(GATES.main(), 1)
            prepare.assert_not_called()
            capture.assert_not_called()
            process.assert_not_called()
            result = json.loads(next((crate / "target/release-gates").glob("*/results.json")).read_text())
            row, = result["gates"]
            self.assertFalse(result["passed"])
            self.assertFalse(row["storage_preflight"]["passed"])
            self.assertEqual(row["storage_preflight"]["free_bytes"], GATES.MINIMUM_BUILD_FREE_BYTES - 1)
            self.assertIsNone(row["process_exit_code"])
            self.assertEqual(row["exit_code"], 1)
            self.assertIn("20 GiB", row["validation_error"])
            self.assertNotIn("compiler_evidence", row)
            self.assertNotIn("coverage", row)

    def test_storage_is_rechecked_before_each_compile_but_not_formatting(self):
        """One passing observation cannot authorize a later compile after space falls."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            commands = [["cargo", "+stable", "fmt", "--check"],
                        ["cargo", "+stable", "test"], ["cargo", "+stable", "doc"]]
            self.disk_usage.side_effect = [SimpleNamespace(free=GATES.MINIMUM_BUILD_FREE_BYTES),
                                          SimpleNamespace(free=GATES.MINIMUM_BUILD_FREE_BYTES - 1)]
            environment = {"CARGO_TARGET_DIR": "/explicit/target"}
            with (
                mock.patch.object(GATES, "CRATE", crate),
                mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "source"]),
                mock.patch.object(GATES, "command_plan", return_value=commands),
                mock.patch.object(GATES.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as process,
                mock.patch.dict(GATES.os.environ, environment, clear=True),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(GATES.main(), 1)
            self.assertEqual(self.disk_usage.call_count, 2)
            self.assertEqual([call.args[0] for call in process.call_args_list], commands[:2])
            self.assertTrue(all(call.kwargs["env"]["CARGO_TARGET_DIR"] == environment["CARGO_TARGET_DIR"]
                                for call in process.call_args_list))
            result = json.loads(next((crate / "target/release-gates").glob("*/results.json")).read_text())
            formatted, compiled, blocked = result["gates"]
            self.assertNotIn("storage_preflight", formatted)
            self.assertTrue(compiled["storage_preflight"]["passed"])
            self.assertEqual(compiled["process_exit_code"], 0)
            self.assertFalse(blocked["storage_preflight"]["passed"])
            self.assertIsNone(blocked["process_exit_code"])
            self.assertFalse(result["passed"])

    def test_unreadable_free_space_is_recorded_without_spawning(self):
        """Failure to measure is distinct from an observed low-space count."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            self.disk_usage.side_effect = OSError("storage query unavailable")
            with (
                mock.patch.object(GATES, "CRATE", crate),
                mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "accelerator-compile"]),
                mock.patch.object(GATES.subprocess, "run") as process,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(GATES.main(), 1)
            process.assert_not_called()
            result = json.loads(next((crate / "target/release-gates").glob("*/results.json")).read_text())
            row, = result["gates"]
            self.assertIsNone(row["storage_preflight"]["free_bytes"])
            self.assertEqual(row["storage_preflight"]["minimum_free_bytes"], GATES.MINIMUM_BUILD_FREE_BYTES)
            self.assertFalse(row["storage_preflight"]["passed"])
            self.assertIn("storage query unavailable", row["validation_error"])
            self.assertIsNone(row["process_exit_code"])

    def test_dry_run_does_not_measure_storage_or_create_evidence(self):
        """Command previews remain read-only even when the host cannot report storage."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            self.disk_usage.side_effect = OSError("not available")
            with (
                mock.patch.object(GATES, "CRATE", crate),
                mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "source", "--dry-run"]),
                mock.patch.object(GATES.subprocess, "run") as process,
                mock.patch("builtins.print"),
            ):
                self.assertEqual(GATES.main(), 0)
            process.assert_not_called()
            self.disk_usage.assert_not_called()
            self.assertFalse((crate / "target").exists())

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

    def test_opt_in_compiler_binding_failure_cannot_adopt_native_success(self):
        """Cache-hit/incomplete evidence fails even when the legacy aggregate passes."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory).resolve()
            minimal_crate(crate)

            def run(command, **kwargs):
                """Emit a passing legacy report while the binding validator rejects it."""

                kwargs["stdout"].write("finished\n")
                output = Path(kwargs["stdout"].name).parent
                (output / "coverage.json").write_text(json.dumps({"data": [{"files": [
                    {"filename": str(crate / "src/lib.rs"), "summary": {"lines": {"count": 1, "covered": 1}}}]}]}))
                return subprocess.CompletedProcess(command, 0)

            with (
                mock.patch.object(GATES, "CRATE", crate),
                mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "coverage", "--coverage-compiler-evidence"]),
                mock.patch.object(GATES, "prepare_capture", side_effect=lambda crate, output, toolchain, features, env: (env, {})),
                mock.patch.object(GATES, "validate_capture", side_effect=ValueError("no fresh compiler captures")),
                mock.patch.object(GATES, "coverage_tool_versions", return_value=[]),
                mock.patch.object(GATES.subprocess, "run", side_effect=run), mock.patch("builtins.print"),
            ):
                self.assertEqual(GATES.main(), 1)
            row = json.loads(next((crate / "target/release-gates").glob("*/results.json")).read_text())["gates"][0]
            self.assertEqual(row["process_exit_code"], 0)
            self.assertEqual(row["exit_code"], 1)
            self.assertNotIn("coverage", row)
            self.assertFalse(row["compiler_evidence"]["artifact_binding_verified"])
            self.assertIn("no fresh", row["validation_error"])

    def test_opt_in_compiler_binding_is_separate_and_does_not_leak_wrappers(self):
        """Successful bindings retain raw counters, while later gates use normal env."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory).resolve()
            minimal_crate(crate)
            observed = []

            def run(command, **kwargs):
                """Emit both representations without invoking native tooling."""

                observed.append((command, kwargs["env"].get("RUSTC_WORKSPACE_WRAPPER")))
                kwargs["stdout"].write("finished\n")
                if "llvm-cov" in command:
                    kwargs["stdout"].write(GATES.UNFILTERED_OPTION_NOTICE + "\n")
                    output = Path(kwargs["stdout"].name).parent
                    report = json.dumps({"data": [{"files": [
                        {"filename": str(crate / "src/lib.rs"), "summary": {"lines": {"count": 1, "covered": 1}}}]}]})
                    (output / "coverage.json").write_text(report)
                    (output / "native.json").write_text(report)
                return subprocess.CompletedProcess(command, 0)

            def validate(setup, snapshot, root):
                """Return only narrow artifact/input claims, never exclusion permission."""

                path = next((crate / "target/release-gates").glob("*/native.json"))
                return {"artifact_binding_verified": True, "compiler_reported_inputs_verified": True,
                        "no_export_filter_verified": True,
                        "compiler_dependency_closure_verified": False, "production_only_exclusions_qualified": False,
                        "native_report": {"path": str(path), "identity": GATES.fingerprint_file(path)}}

            with (
                mock.patch.object(GATES, "CRATE", crate),
                mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "coverage", "--gate", "source", "--coverage-compiler-evidence"]),
                mock.patch.object(GATES, "prepare_capture", side_effect=lambda crate, output, toolchain, features, env: ({**env, "RUSTC_WORKSPACE_WRAPPER": "run-local"}, {})),
                mock.patch.object(GATES, "validate_capture", side_effect=validate),
                mock.patch.object(GATES, "coverage_tool_versions", return_value=[]),
                mock.patch.object(GATES.subprocess, "run", side_effect=run), mock.patch("builtins.print"),
                mock.patch.dict(GATES.os.environ, {}, clear=True),
            ):
                self.assertEqual(GATES.main(), 0)
            result = json.loads(next((crate / "target/release-gates").glob("*/results.json")).read_text())
            self.assertTrue(result["passed"])
            row = result["gates"][0]
            self.assertTrue(row["compiler_evidence"]["artifact_binding_verified"])
            self.assertTrue(row["storage_preflight"]["passed"])
            self.assertFalse(row["compiler_evidence"]["production_only_exclusions_qualified"])
            self.assertEqual(observed[0][1], "run-local")
            self.assertTrue(all(wrapper is None for _, wrapper in observed[1:]))

    def test_only_one_exact_reviewed_option_notice_is_permitted_when_opted_in(self):
        """The non-numerical upstream option notice never hides a compiler/LLVM warning."""

        notice = GATES.UNFILTERED_OPTION_NOTICE
        self.assertEqual(GATES.require_clean_coverage_log(notice + "\n", allow_unfiltered_option_notice=True), 1)
        with self.assertRaises(ValueError):
            GATES.require_clean_coverage_log(notice + "\n")
        for log in ("", notice + "\n" + notice, notice + "\nwarning: mismatched data", notice + "\nwarning[unused]: test",
                    " " + notice, notice + " ", notice + "\r\n", "\x1b[33m" + notice + "\x1b[0m"):
            with self.subTest(log=log), self.assertRaises(ValueError):
                GATES.require_clean_coverage_log(log, allow_unfiltered_option_notice=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.log"
            path.write_bytes((notice + "\r\n").encode())
            self.assertEqual(GATES.read_coverage_log(path), notice + "\r\n")
            with self.assertRaises(ValueError):
                GATES.require_clean_coverage_log(GATES.read_coverage_log(path), allow_unfiltered_option_notice=True)

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
                minimal_crate(crate)

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
                    mock.patch.object(GATES, "coverage_tool_versions", return_value=[]),
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

    def run_coverage_fixture(self, crate, callback):
        """Run the real runner around one fake native coverage subprocess."""

        with (
            mock.patch.object(GATES, "CRATE", crate),
            mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "coverage"]),
            mock.patch.object(GATES.subprocess, "run", side_effect=callback) as process,
            mock.patch.object(GATES, "coverage_tool_versions", return_value=[]),
            mock.patch("builtins.print"),
        ):
            status = GATES.main()
        manifest, = (crate / "target/release-gates").glob("*/results.json")
        return status, json.loads(manifest.read_text()), manifest.parent, process.call_count

    def passing_export(self, command, **kwargs):
        """Emit a well-formed perfect-coverage fixture and a clean native log."""

        kwargs["stdout"].write("finished\n")
        path = Path(kwargs["stdout"].name).parent / "coverage.json"
        path.write_text(json.dumps({"data": [{"files": [file_entry("src/lib.rs", 100, 100)]}]}))
        return subprocess.CompletedProcess(command, 0)

    def test_stable_coverage_records_both_snapshots_and_raw_identities(self):
        """A green native run is byte-bound but still not production-qualified."""

        with tempfile.TemporaryDirectory() as temporary:
            crate = Path(temporary)
            minimal_crate(crate)
            status, result, output, calls = self.run_coverage_fixture(crate, self.passing_export)
            self.assertEqual((status, calls), (0, 1))
            self.assertTrue(result["passed"])
            row, = result["gates"]
            self.assertTrue(row["storage_preflight"]["passed"])
            evidence = row["input_evidence"]
            self.assertTrue(evidence["sources_unchanged"])
            self.assertFalse(evidence["production_only_exclusions_qualified"])
            before = json.loads((output / evidence["before"]).read_text())
            after = json.loads((output / evidence["after"]).read_text())
            self.assertEqual(before, after)
            self.assertFalse(before["compiler_dependency_closure_verified"])
            for name, key in (("coverage.json", "report_identity"), (row["log"], "log_identity")):
                self.assertEqual(row[key]["sha256"], hashlib.sha256((output / name).read_bytes()).hexdigest())

    def test_source_mutation_fails_before_any_counter_is_adopted(self):
        """Even same-size input edits invalidate otherwise passing native output."""

        with tempfile.TemporaryDirectory() as temporary:
            crate = Path(temporary)
            minimal_crate(crate)

            def mutate(command, **kwargs):
                """Change the source while the synthetic native command runs."""

                (crate / "src/lib.rs").write_text("pub fn value() -> u32 { 8 }\n")
                return self.passing_export(command, **kwargs)

            with mock.patch.object(GATES, "coverage_summary") as summary:
                status, result, output, _ = self.run_coverage_fixture(crate, mutate)
            summary.assert_not_called()
            self.assertEqual(status, 1)
            row, = result["gates"]
            self.assertEqual(row["process_exit_code"], 0)
            self.assertIn("src/lib.rs", row["validation_error"])
            self.assertFalse(row["input_evidence"]["sources_unchanged"])
            self.assertTrue((output / "coverage-inputs-after.json").is_file())

    def test_uncaptured_or_non_rust_coverage_entries_fail_before_adopting_counters(self):
        """Stable source files do not make a stale deleted-file export acceptable."""

        for filename in ("src/deleted.rs", "src/fixture.txt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                crate = Path(temporary)
                minimal_crate(crate)
                (crate / "src/fixture.txt").write_text("fixture\n")

                def stale(command, **kwargs):
                    """Emit passing counters for one file outside the Rust inventory."""

                    kwargs["stdout"].write("finished\n")
                    path = Path(kwargs["stdout"].name).parent / "coverage.json"
                    path.write_text(json.dumps({"data": [{"files": [file_entry(filename, 100, 100)]}]}))
                    return subprocess.CompletedProcess(command, 0)

                with mock.patch.object(GATES, "coverage_summary") as summary:
                    status, result, _, _ = self.run_coverage_fixture(crate, stale)
                summary.assert_not_called()
                self.assertEqual(status, 1)
                self.assertIn("not a captured Rust input", result["gates"][0]["validation_error"])

    def test_relative_llvm_source_paths_use_the_compiler_working_directory(self):
        """Invoking the runner from a parent checkout must not silently drop files."""

        report = {"data": [{"files": [{"filename": "src/lib.rs", "summary": {"lines": {"count": 100, "covered": 100}}}]}]}
        with tempfile.TemporaryDirectory() as temporary:
            crate = Path(temporary)
            minimal_crate(crate)
            GATES.require_bound_coverage_sources(report, GATES.capture_sources(crate), crate)
            self.assertTrue(GATES.coverage_summary(report, crate)["passed"])
            for invalid in (None, "", "src/\0.rs"):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    GATES.coverage_source_relative(invalid, crate)

    def test_native_failure_still_records_terminal_inputs_without_using_counters(self):
        """Preserve native failure separately from any validation outcome."""

        with tempfile.TemporaryDirectory() as temporary:
            crate = Path(temporary)
            minimal_crate(crate)

            def fail(command, **kwargs):
                """Write a report, then report a failing native command."""

                self.passing_export(command, **kwargs)
                return subprocess.CompletedProcess(command, 7)

            status, result, output, _ = self.run_coverage_fixture(crate, fail)
            self.assertEqual(status, 1)
            row, = result["gates"]
            self.assertEqual((row["process_exit_code"], row["exit_code"]), (7, 7))
            self.assertNotIn("coverage", row)
            self.assertTrue((output / "coverage-inputs-after.json").is_file())

    def test_failed_preflight_or_spawn_leaves_a_failed_manifest(self):
        """Missing inputs and tools must not leave an unrecorded successful prefix."""

        for inputs in (False, True):
            with self.subTest(inputs=inputs), tempfile.TemporaryDirectory() as temporary:
                crate = Path(temporary)
                if inputs:
                    minimal_crate(crate)

                def missing(*args, **kwargs):
                    """Model a missing Cargo executable after a valid snapshot."""

                    raise FileNotFoundError("cargo missing")

                status, result, _, calls = self.run_coverage_fixture(crate, missing)
                row, = result["gates"]
                self.assertEqual((status, calls), (1, int(inputs)))
                self.assertFalse(result["passed"])
                self.assertIsNone(row["process_exit_code"])
                self.assertIn("cargo missing" if inputs else "mandatory", row["validation_error"])

    def test_report_or_log_edit_during_validation_discards_the_summary(self):
        """Hashing raw evidence once does not excuse a later parse-time change."""

        original = GATES.coverage_summary
        for filename in ("coverage.json", "00-coverage.log"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                crate = Path(temporary)
                minimal_crate(crate)

                def edit(report, root):
                    """Change the retained artifact after counters are parsed."""

                    result = original(report, root)
                    path, = (crate / "target/release-gates").glob(f"*/{filename}")
                    path.write_text(path.read_text() + "\n")
                    return result

                with mock.patch.object(GATES, "coverage_summary", side_effect=edit):
                    status, result, _, _ = self.run_coverage_fixture(crate, self.passing_export)
                self.assertEqual(status, 1)
                row, = result["gates"]
                self.assertIn("changed during validation", row["validation_error"])
                self.assertNotIn("coverage", row)

    def test_context_changes_are_rejected_and_tool_versions_are_explicit(self):
        """Environment, runner and selected tools are evidence, not hidden defaults."""

        command = ["cargo", "+stable", "llvm-cov"]
        with mock.patch.object(GATES.subprocess, "check_output", return_value="tool version\n") as run:
            versions = GATES.coverage_tool_versions(command, GATES.CRATE, {})
        self.assertEqual(len(versions), 3)
        self.assertTrue(all(call.args[0][1] == "+stable" for call in run.call_args_list))
        for output in ("", "warning: stale executable\n"):
            with mock.patch.object(GATES.subprocess, "check_output", return_value=output), self.assertRaises(ValueError):
                GATES.coverage_tool_versions(command, GATES.CRATE, {})
        with self.assertRaises(ValueError):
            GATES.coverage_tool_versions(["cargo", "llvm-cov"], GATES.CRATE, {})
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(GATES, "coverage_tool_versions", return_value=[]):
            crate = Path(temporary)
            minimal_crate(crate)
            before = GATES.capture_coverage_context(crate, command, {"CARGO_INCREMENTAL": "0", "SECRET_TOKEN": "not recorded"})
            self.assertNotIn("SECRET_TOKEN", before["compiler_environment"])
            for key in ("runner_inputs", "tool_versions", "command", "working_directory", "compiler_environment"):
                after = {**before, key: None}
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    GATES.require_same_coverage_context(before, after)

    def test_duplicate_gate_names_fail_before_creating_artifacts(self):
        """Repeated coverage groups must not overwrite earlier raw evidence."""

        with mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "coverage", "--gate", "coverage"]), self.assertRaisesRegex(ValueError, "duplicate gate"):
            GATES.main()

    def test_all_command_groups_are_scoped(self):
        """Build every group without shell interpolation or mutable lockfiles."""

        for gate in GATES.GATES:
            with self.subTest(gate=gate):
                commands = GATES.command_plan(gate, Path("/tmp/output with spaces"), "stable", "1.92.0", True,
                                              verus="/opt/verus/verus")
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
        """Only the reviewed IDs are waived lockfile-wide; stale DBs still fail."""

        command = GATES.command_plan("dependencies", Path("/tmp/output"), "stable", "1.92.0", True)[0]
        self.assertIn("--no-fetch", command)
        self.assertIn("--no-yanked", command)
        self.assertNotIn("--stale", command)
        waived = [command[index + 1] for index, part in enumerate(command) if part == "--ignore"]
        self.assertEqual(waived, list(GATES.MAINTENANCE_NOTICES + GATES.UNREACHABLE_ADVISORIES))
        self.assertIn("--deny", command)

    def test_unreachable_vulnerabilities_stay_denied_in_the_built_graph(self):
        """Only cargo-audit waives unreachable vulnerabilities; deny.toml and CI agree."""

        policy = (GATES.CRATE / "deny.toml").read_text()
        workflow = (GATES.CRATE / ".github/workflows/ci.yml").read_text()
        for advisory in GATES.UNREACHABLE_ADVISORIES:
            self.assertNotIn(advisory, policy)
            self.assertIn(f"--ignore {advisory}", workflow)
        for advisory in GATES.MAINTENANCE_NOTICES:
            self.assertIn(advisory, policy)
            self.assertIn(f"--ignore {advisory}", workflow)

    def test_built_graph_advisories_include_yanked_checks(self):
        """cargo-deny judges advisories over the built graph, offline without fetching."""

        for offline in (False, True):
            deny = GATES.command_plan("dependencies", Path("/tmp/output"), "stable", "1.92.0", offline)[1]
            self.assertEqual(deny[-3:], ["advisories", "licenses", "sources"])
            self.assertEqual("--offline" in deny, offline)
            audit = GATES.command_plan("dependencies", Path("/tmp/output"), "stable", "1.92.0", offline)[0]
            self.assertIn("--no-yanked", audit)
        policy = (GATES.CRATE / "deny.toml").read_text()
        self.assertIn('yanked = "deny"', policy)
        for advisory in GATES.MAINTENANCE_NOTICES:
            self.assertIn(advisory, policy)

    def test_source_experiments_run_without_checkpoint_or_network_preparation(self):
        """Compilation alone cannot validate the public experiments' assertions."""

        commands = GATES.command_plan("source", Path("/tmp/output"), "stable", "1.92.0", True)
        crate_tests = [command for command in commands
                       if command[:3] == ["cargo", "+stable", "test"]
                       and str(GATES.CRATE / "Cargo.toml") in command]
        self.assertTrue(any("--all-targets" in command for command in crate_tests))
        self.assertTrue(any("--doc" in command for command in crate_tests))
        runs = [command for command in commands if command[:3] == ["cargo", "+stable", "run"]]
        names = {command[command.index("--example") + 1] for command in runs}
        self.assertEqual(names, {
            "activation_workbench", "attention_head_attribution", "custom_composition",
            "t5_classifier", "mla_encoder_decoder", "qwen_prefix", "burn_training",
        })
        for command in runs:
            self.assertIn("--release", command)
            self.assertIn("--offline", command)
            self.assertIn("--locked", command)
            self.assertIn("flex", command[command.index("--features") + 1].split(","))
            self.assertIn("train", command[command.index("--features") + 1].split(","))
            if command[command.index("--example") + 1] == "qwen_prefix":
                self.assertEqual(command[-2:], ["--", "--random"])
            if command[command.index("--example") + 1] == "burn_training":
                self.assertEqual(command[-2:], ["--", "/tmp/output/burn_training"])
        self.assertFalse(any("prepare.py" in argument for command in commands for argument in command))

    def test_source_checks_compile_native_training_without_changing_coverage_profile(self):
        """The gated Learner example must not disappear behind an inactive cfg."""

        commands = GATES.command_plan("source", Path("/tmp/output"), "stable", "1.92.0", True)
        selected = [command for command in commands
                    if command[0] == "cargo" and str(GATES.CRATE / "Cargo.toml") in command
                    and command[2] in {"clippy", "test", "doc"}]
        self.assertEqual({command[2] for command in selected}, {"clippy", "test", "doc"})
        for command in selected:
            self.assertIn("train", command[command.index("--features") + 1].split(","))
        coverage = GATES.command_plan("coverage", Path("/tmp/output"), "stable", "1.92.0", True)[0]
        self.assertEqual(coverage[coverage.index("--features") + 1], GATES.FEATURES)

    def test_standalone_classifier_has_separate_source_and_msrv_checks(self):
        """The dev-only parser has its own lockfile, tests and warning-denied lint."""

        for gate, expected_count in (("source", 4), ("msrv", 1)):
            commands = GATES.command_plan(gate, Path("/tmp/output"), "stable", "1.92.0", True)
            selected = [command for command in commands if str(GATES.CRATE / "scripts/coverage-layout/Cargo.toml") in command]
            self.assertEqual(len(selected), expected_count)
            for command in selected:
                if command[2] != "fmt":
                    self.assertIn("--locked", command)
                    self.assertIn("--offline", command)
                    self.assertEqual(command[command.index("--target-dir") + 1], str(GATES.CRATE / "target"))
                if gate == "msrv":
                    self.assertEqual(command[1], "+1.92.0")
            if gate == "source":
                self.assertEqual([command[2] for command in selected], ["fmt", "clippy", "test", "build"])
                audit, = [command for command in commands if str(GATES.CRATE / "scripts/check_documentation.py") in command]
                self.assertGreater(commands.index(audit), commands.index(selected[-1]))
                self.assertEqual(audit[audit.index("--classifier") + 1], str(GATES.CRATE / "target/debug/transformers-coverage-layout"))
                self.assertEqual(audit[audit.index("--output") + 1], "/tmp/output/documentation.json")
                self.assertIn("--standards", audit)
                self.assertNotIn("--inventory-only", audit)

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
        """Unanchored filename globs also select artifacts through model symlinks.

        Exclusions (``!`` patterns) are anchored the same way, so each removes
        exactly the repository paths it names.
        """

        manifest = tomllib.loads((GATES.CRATE / "Cargo.toml").read_text())
        self.assertTrue(all(pattern.removeprefix("!").startswith("/") for pattern in manifest["package"]["include"]))
        self.assertIn("/scripts/COVERAGE_EVIDENCE.md", manifest["package"]["include"])
        self.assertTrue((GATES.CRATE / "scripts/COVERAGE_EVIDENCE.md").is_file())
        self.assertIn("/scripts/WHOLE_FILE_COVERAGE.md", manifest["package"]["include"])
        self.assertTrue((GATES.CRATE / "scripts/WHOLE_FILE_COVERAGE.md").is_file())

    def test_package_ships_the_files_its_verus_harness_test_reads(self):
        """tests/verus_harness.rs is packaged, so the harnesses and their guide must be too."""

        include = tomllib.loads((GATES.CRATE / "Cargo.toml").read_text())["package"]["include"]
        self.assertIn("/tests/**", include)
        self.assertIn("/verus/*.rs", include)
        self.assertIn("/verus/README.md", include)
        base = "Cargo.toml\nCargo.lock\nREADME.md\nsrc/lib.rs\n"
        self.assertEqual(GATES.validate_package_listing(base + "verus/cache.rs\nverus/README.md\n"), 6)
        for entry in ("verus/nested/cache.rs", "verus/notes.txt", "verus/.verus-log/root.air"):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, "verus/"):
                GATES.validate_package_listing(base + entry + "\n")

    def test_package_gate_runs_the_archive_test_suite(self):
        """Cargo's verification only builds the archive; the gate also runs its tests.

        The tests reuse Cargo's normal target directory, as package verification
        does: without it Cargo would build every dependency again inside the
        extracted archive, which the next ``cargo package`` deletes.
        """

        # Read the stem from the manifest so a version bump or a package
        # rename does not break this test; test_package_stem_is_name_and_version
        # pins the format itself.
        package_table = tomllib.loads((GATES.CRATE / "Cargo.toml").read_text())["package"]
        stem = f"{package_table['name']}-{package_table['version']}"
        for target, expected in ((None, GATES.CRATE / "target"), ("/explicit/target", Path("/explicit/target")),
                                 ("relative-target", GATES.CRATE / "relative-target")):
            environment = {} if target is None else {"CARGO_TARGET_DIR": target}
            with self.subTest(target=target), mock.patch.dict(GATES.os.environ, environment, clear=True):
                listing, package, archive_tests, archive_scripts = GATES.command_plan(
                    "package", Path("/tmp/output"), "stable", "1.92.0", True)
                archive = GATES.package_archive()
            self.assertIn("--list", listing)
            self.assertEqual(package[:3], ["cargo", "+stable", "package"])
            self.assertEqual(archive, expected / "package" / f"{stem}.crate")
            self.assertEqual(archive_tests[:3], ["cargo", "+stable", "test"])
            manifest = Path(archive_tests[archive_tests.index("--manifest-path") + 1])
            self.assertEqual(manifest, expected / "package" / stem / "Cargo.toml")
            self.assertEqual(archive_tests.count("--target-dir"), 1)
            self.assertEqual(Path(archive_tests[archive_tests.index("--target-dir") + 1]), expected)
            self.assertIn("--locked", archive_tests)
            self.assertIn("--offline", archive_tests)
            self.assertTrue(GATES.command_compiles(archive_tests))
            # The packaged Python tests run from the archive, not the checkout,
            # so a record test that needs repository-only evidence must skip.
            self.assertEqual(archive_scripts[:4], [GATES.sys.executable, "-m", "unittest", "discover"])
            self.assertEqual(Path(archive_scripts[archive_scripts.index("-s") + 1]),
                             expected / "package" / stem / "scripts/tests")
            self.assertFalse(GATES.command_compiles(archive_scripts))

    def test_package_stem_is_name_and_version(self):
        """Cargo names the archive and its extraction NAME-VERSION from [package]."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            (crate / "Cargo.toml").write_text('[package]\nname = "renamed-crate"\nversion = "2.3.4-rc.1"\n\n'
                                              '[lib]\nname = "transformers"\n')
            with mock.patch.object(GATES, "CRATE", crate), \
                    mock.patch.dict(GATES.os.environ, {"CARGO_TARGET_DIR": "/explicit/target"}, clear=True):
                self.assertEqual(GATES.package_stem(), "renamed-crate-2.3.4-rc.1")
                self.assertEqual(GATES.package_archive(), Path("/explicit/target/package/renamed-crate-2.3.4-rc.1.crate"))
                self.assertEqual(GATES.package_directory(), Path("/explicit/target/package/renamed-crate-2.3.4-rc.1"))

    def test_package_include_matches_the_evidence_policy(self):
        """Cargo's include list publishes exactly the evidence package_contents.py allows."""

        include = tomllib.loads((GATES.CRATE / "Cargo.toml").read_text())["package"]["include"]
        acceptance = {pattern for pattern in include if pattern.removeprefix("!").startswith("/acceptance")}
        self.assertEqual(acceptance, {
            "/acceptance/*.json", "/acceptance/measurements/*.json", "/acceptance/measurements/README.md",
            *(f"/acceptance/measurements/{campaign}/**" for campaign in GATES.package_contents.PUBLISHED_CAMPAIGNS),
        })
        self.assertIn("/examples/**", include)
        self.assertEqual({pattern for pattern in include if pattern.startswith("!")}, {
            "!/examples/mechanistic/**/*.png", "!/examples/mechanistic/**/*.svg",
            "!/examples/mechanistic/**/result.json", "!/examples/mechanistic/**/*.result.json",
            "!/examples/mechanistic/*/evidence/**", "!/examples/mechanistic/*/results/**",
        })
        for campaign in GATES.package_contents.PUBLISHED_CAMPAIGNS:
            if not GATES.package_contents.is_packaged_archive(GATES.CRATE):
                self.assertTrue((GATES.CRATE / "acceptance/measurements" / campaign).is_dir(), campaign)

    def test_package_listing_rejects_repository_only_evidence(self):
        """Unpublished campaign directories and study outputs fail the listing check."""

        base = "Cargo.toml\nCargo.lock\nREADME.md\nsrc/lib.rs\n"
        published = ("acceptance/qwen3.5-0.8b-2fc06364715b967f1860aea9cf38778875588b17.json",
                     "acceptance/measurements/v0_1-release-20260910.json",
                     "acceptance/measurements/README.md",
                     "acceptance/measurements/small-model-performance-20260910/manifest.json",
                     "examples/mechanistic/greater_than/manifest.json",
                     "examples/mechanistic/greater_than/fixture.json",
                     "examples/mechanistic/grokking_training/study.rs",
                     "examples/mechanistic/tuned_lens/plot.py")
        for entry in published:
            with self.subTest(entry=entry):
                self.assertEqual(GATES.validate_package_listing(base + entry + "\n"), 5)
        for entry in ("acceptance/measurements/coding-standards-coverage-20260910/final/coverage.json",
                      "acceptance/measurements/large-checkpoint-replay-20261016/gates.log",
                      "acceptance/measurements/public-examples-20260909/Cargo.lock",
                      "acceptance/measurements/notes.txt",
                      "acceptance/history/record.json",
                      "acceptance/README.md",
                      "examples/mechanistic/greater_than/result.json",
                      "examples/mechanistic/v0_2_examples.result.json",
                      "examples/mechanistic/tuned_lens/prediction_quality.png",
                      "examples/mechanistic/grokking_training/training.svg",
                      "examples/mechanistic/grokking_training/evidence/run.json",
                      "examples/mechanistic/clock_and_pizza_training/results/alpha0-paper/protocol.json"):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, "repository-only evidence"):
                GATES.validate_package_listing(base + entry + "\n")

    def run_package_fixture(self, crate, archive_bytes):
        """Run the real package gate over a fixture crate whose archive has ``archive_bytes``.

        Cargo is replaced by a stand-in that prints a valid listing and, for the
        verified ``cargo package``, writes a sparse archive of the given size.
        ``None`` writes no archive at all.
        """

        def run(command, **kwargs):
            """Print the listing or create the archive that the real command would."""

            if "--list" in command:
                kwargs["stdout"].write("Cargo.toml\nCargo.lock\nREADME.md\nsrc/lib.rs\n")
            elif command[2:3] == ["package"] and archive_bytes is not None:
                archive = crate / "target/package/fixture-0.1.0.crate"
                archive.parent.mkdir(parents=True, exist_ok=True)
                with archive.open("wb") as handle:
                    handle.truncate(archive_bytes)
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch.object(GATES, "CRATE", crate),
            mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "package", "--offline"]),
            mock.patch.dict(GATES.os.environ, {}, clear=True),
            mock.patch.object(GATES.subprocess, "run", side_effect=run),
            mock.patch("builtins.print"),
        ):
            status = GATES.main()
        manifest, = (crate / "target/release-gates").glob("*/results.json")
        return status, json.loads(manifest.read_text())

    def test_package_gate_records_the_archive_within_its_budget(self):
        """The verified archive's size and SHA-256 are kept; an oversized one fails."""

        for archive_bytes, passed in ((1024, True), (GATES.PACKAGE_MAX_BYTES, True),
                                      (GATES.PACKAGE_MAX_BYTES + 1, False), (None, False)):
            with self.subTest(archive_bytes=archive_bytes), tempfile.TemporaryDirectory() as directory:
                crate = Path(directory)
                (crate / "Cargo.toml").write_text('[package]\nname = "fixture"\nversion = "0.1.0"\n')
                status, result = self.run_package_fixture(crate, archive_bytes)
                self.assertEqual((status, result["passed"]), (0 if passed else 1, passed))
                listing, package = result["gates"][:2]
                self.assertEqual(listing["package_files"], 4)
                if passed:
                    self.assertEqual(len(result["gates"]), 4)
                    archive = package["package_archive"]
                    self.assertEqual(archive["bytes"], archive_bytes)
                    self.assertEqual(archive["maximum_bytes"], GATES.PACKAGE_MAX_BYTES)
                    expected = hashlib.sha256(b"\0" * archive_bytes).hexdigest()
                    self.assertEqual(archive["sha256"], expected)
                    self.assertEqual(Path(archive["path"]), crate / "target/package/fixture-0.1.0.crate")
                else:
                    self.assertEqual(len(result["gates"]), 2)
                    self.assertEqual(package["process_exit_code"], 0)
                    self.assertNotIn("package_archive", package)
                    self.assertRegex(package["validation_error"],
                                     "budget below crates.io" if archive_bytes else "not a regular file")

    def test_guide_fragments_compile_inside_their_hidden_context(self):
        """A marked fragment is wrapped in its context as rustdoc hidden lines."""

        markdown = (
            "# Guide\n\n<!-- doctest-context model\nuse burn::tensor::backend::Backend;\n\n"
            "fn fragment<B: Backend>(value: u8) {\n    __FRAGMENT__\n}\n-->\n"
            "Prose.\n\n<!-- doctest model -->\n```rust\nlet doubled = value * 2;\n```\n"
        )
        rewritten, compiled = GATES.guide_doctests(markdown)
        self.assertEqual(compiled, 1)
        self.assertNotIn("doctest-context", rewritten)
        self.assertIn(
            "```rust\n# use burn::tensor::backend::Backend;\n#\n# fn fragment<B: Backend>(value: u8) {\n"
            "let doubled = value * 2;\n# }\n```", rewritten)
        self.assertTrue(rewritten.startswith("# Guide\n\nProse."))

    def test_guide_blocks_outside_markers_keep_their_language(self):
        """Complete Rust compiles as written, other languages never, and bare fences become text."""

        markdown = ("```rust\nfn complete() {}\n```\n\n```sh\ncargo test\n```\n\n```\nplain\n```\n\n"
                    "```rust,ignore\nnot compiled\n```\n\n- item\n\n  ```rust\n  fn nested() {}\n  ```\n")
        rewritten, compiled = GATES.guide_doctests(markdown)
        self.assertEqual(compiled, 2)
        self.assertIn("```rust\nfn complete() {}\n```", rewritten)
        self.assertIn("```sh\ncargo test\n```", rewritten)
        self.assertIn("```text\nplain\n```", rewritten)
        self.assertIn("```rust,ignore\nnot compiled\n```", rewritten)
        self.assertIn("  ```rust\n  fn nested() {}\n  ```", rewritten)

    def test_guide_doctest_markers_fail_closed(self):
        """Unknown, unused, duplicate, or malformed contexts cannot hide a fragment."""

        context = "<!-- doctest-context a\nfn f() {\n__FRAGMENT__\n}\n-->\n"
        fragment = "<!-- doctest a -->\n```rust\nlet x = 1;\n```\n"
        for markdown, message in (
            ("<!-- doctest missing -->\n```rust\nlet x = 1;\n```\n", "unknown"),
            (context, "unused"),
            (context + context + fragment, "unique"),
            ("<!-- doctest-context a\nfn f() {}\n-->\n" + fragment, "__FRAGMENT__"),
            (context + "<!-- doctest a -->\n```sh\nls\n```\n", "non-Rust"),
            (context + "<!-- doctest a -->\n```rust\nlet x = 1;\n", "unclosed"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                GATES.guide_doctests(markdown)

    def test_repository_guides_mark_every_fragment_consistently(self):
        """Every Rust block of every gated guide is compiled; markers and contexts match."""

        for guide in GATES.GUIDES:
            with self.subTest(guide=guide):
                text = (GATES.CRATE / guide).read_text()
                _, compiled = GATES.guide_doctests(text)
                blocks = [line for line in text.split("\n") if line.strip().startswith("```rust")]
                self.assertEqual(compiled, len([block for block in blocks if "ignore" not in block]))
                self.assertGreater(compiled, 0)

    def test_guide_doctest_crate_depends_on_this_crate_by_path(self):
        """The scratch crate uses the source gate's features and this crate's lockfile."""

        for package_name in ("transformers", "renamed-transformers"):
            with self.subTest(package=package_name):
                self.check_guide_doctest_crate(package_name)

    def check_guide_doctest_crate(self, package_name):
        """Generate the guide doctest crate for a crate whose package is ``package_name``.

        The guides import the library as ``transformers``, so the dependency
        keeps that key and names the package explicitly; a renamed package
        with ``[lib] name = "transformers"`` then needs no tooling change.
        """

        with tempfile.TemporaryDirectory() as directory:
            crate, scratch = Path(directory) / "crate", Path(directory) / "out/guides"
            crate.mkdir()
            (crate / "Cargo.toml").write_text(f'[package]\nname = "{package_name}"\n\n[dependencies]\n'
                                             'burn = { version = "=0.21.0", default-features = false }\n')
            (crate / "Cargo.lock").write_text("version = 4\n")
            for guide in GATES.GUIDES:
                (crate / guide).write_text("```rust\nfn complete() {}\n```\n")
            with mock.patch.object(GATES.subprocess, "run") as run:
                summary = GATES.write_guide_doctests(crate, scratch, True)
            self.assertEqual(summary["compiled_blocks"], len(GATES.GUIDES))
            manifest = tomllib.loads((scratch / "Cargo.toml").read_text())
            dependency = manifest["dependencies"]["transformers"]
            self.assertEqual(dependency["package"], package_name)
            self.assertEqual(dependency["path"], str(crate.resolve()))
            self.assertFalse(dependency["default-features"])
            self.assertEqual(dependency["features"], [*GATES.FEATURES.split(","), "train"])
            self.assertEqual(manifest["dependencies"]["burn"]["version"], "=0.21.0")
            self.assertIn("workspace", manifest)
            self.assertEqual((scratch / "Cargo.lock").read_text(), "version = 4\n")
            library = (scratch / "lib.rs").read_text()
            for guide in GATES.GUIDES:
                self.assertIn(f'include_str!("guides/{guide}")', library)
                self.assertTrue((scratch / "guides" / guide).is_file())
            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["cargo", "metadata"])
            self.assertIn("--offline", command)
            self.assertNotIn("--locked", command)

    def test_guides_gate_tests_the_generated_crate_with_the_shared_target(self):
        """The doctest command is locked and reuses Cargo's normal target directory."""

        with mock.patch.dict(GATES.os.environ, {"CARGO_TARGET_DIR": "/explicit/target"}, clear=True):
            command, = GATES.command_plan("guides", Path("/tmp/output"), "stable", "1.92.0", True)
        self.assertEqual(command[:4], ["cargo", "+stable", "test", "--doc"])
        self.assertEqual(command[command.index("--manifest-path") + 1], "/tmp/output/guide-doctests/Cargo.toml")
        self.assertEqual(command[command.index("--target-dir") + 1], "/explicit/target")
        self.assertIn("--locked", command)
        self.assertIn("--offline", command)
        self.assertTrue(GATES.command_compiles(command))

    def test_guides_gate_writes_the_crate_before_running_its_doctests(self):
        """The runner prepares the scratch crate, records its block counts, then runs cargo."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            observed = []

            def run(command, **kwargs):
                """Observe that the scratch crate exists when the doctests run."""

                observed.append(Path(command[command.index("--manifest-path") + 1]).is_file())
                return subprocess.CompletedProcess(command, 0)

            with (
                mock.patch.object(GATES, "CRATE", crate),
                mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "guides"]),
                mock.patch.object(GATES, "write_guide_doctests", side_effect=lambda crate, directory, offline: (
                    directory.mkdir(parents=True), (directory / "Cargo.toml").write_text(""),
                    {"guides": {}, "compiled_blocks": 25})[-1]) as write,
                mock.patch.object(GATES.subprocess, "run", side_effect=run),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(GATES.main(), 0)
            self.assertEqual(observed, [True])
            write.assert_called_once()
            result = json.loads(next((crate / "target/release-gates").glob("*/results.json")).read_text())
            self.assertEqual(result["gates"][0]["guide_doctests"]["compiled_blocks"], 25)

    def test_verus_gate_requires_the_executable(self):
        """An unset VERUS fails before any evidence directory exists."""

        with self.assertRaisesRegex(ValueError, "VERUS"):
            GATES.command_plan("verus", Path("/tmp/output"), "stable", "1.92.0", True)
        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            with (
                mock.patch.object(GATES, "CRATE", crate),
                mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "verus"]),
                mock.patch.dict(GATES.os.environ, {}, clear=True),
                mock.patch.object(GATES.subprocess, "run") as process,
                self.assertRaisesRegex(ValueError, "VERUS"),
            ):
                GATES.main()
            process.assert_not_called()
            self.assertFalse((crate / "target").exists())

    def test_verus_gate_checks_the_version_then_every_harness_without_cheating(self):
        """One version probe, then each verus/*.rs in order with the pinned flags."""

        commands = GATES.command_plan("verus", Path("/tmp/output"), "stable", "1.92.0", True, verus="/opt/verus")
        self.assertEqual(commands[0], ["/opt/verus", "--version"])
        harnesses = sorted((GATES.CRATE / "verus").glob("*.rs"))
        self.assertEqual(len(commands), 1 + len(harnesses))
        for command, harness in zip(commands[1:], harnesses):
            self.assertEqual(command, ["/opt/verus", "--crate-type", "lib", "--no-cheating",
                                       "--triggers-mode", "silent", str(harness)])
            self.assertFalse(GATES.command_compiles(command))

    def test_verus_result_lines_must_report_zero_errors(self):
        """Only one clean result line is a pass; errors and missing results fail."""

        self.assertEqual(GATES.parse_verus_result("note: x\nverification results:: 199 verified, 0 errors\n"), 199)
        self.assertEqual(GATES.parse_verus_result("verification results:: 3 verified, 0 errors"), 3)
        for log in ("", "verification results:: 198 verified, 1 errors\n", "verification results:: 2 verified, 0 errors\n" * 2,
                    "error: aborting due to 1 previous error\n"):
            with self.subTest(log=log), self.assertRaises(ValueError):
                GATES.parse_verus_result(log)
        GATES.require_pinned_verus(VERUS_VERSION_OUTPUT)
        for output in ("", VERUS_VERSION_OUTPUT.replace("426d8b0", "1234567"), "Version: 0.2026.10.04.426d8b0-dirty\n"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "pinned"):
                GATES.require_pinned_verus(output)

    def test_recorded_verus_counts_cover_every_repository_harness(self):
        """The real verus/README.md table names each harness once and sums to its total."""

        counts, total = GATES.recorded_verus_counts((GATES.CRATE / "verus/README.md").read_text())
        self.assertEqual(sorted(counts), [path.name for path in sorted((GATES.CRATE / "verus").glob("*.rs"))])
        self.assertEqual(sum(counts.values()), total)
        for readme in ("# Proofs\n", "## Harnesses\n| `a.rs` | x | many | y |\n**1 verified, 0 errors**\n",
                       "## Harnesses\n| `a.rs` | x | 1 | y |\n| `a.rs` | x | 1 | y |\n**2 verified, 0 errors**\n",
                       "## Harnesses\n| `a.rs` | x |\n**1 verified, 0 errors**\n",
                       "## Harnesses\n| `a.rs` | x | 1 | y |\n"):
            with self.subTest(readme=readme), self.assertRaises(ValueError):
                GATES.recorded_verus_counts(readme)

    def run_verus_fixture(self, crate, run):
        """Run the real runner over a fixture crate with a fake Verus executable."""

        with (
            mock.patch.object(GATES, "CRATE", crate),
            mock.patch.object(GATES.sys, "argv", ["check_release", "--gate", "verus"]),
            mock.patch.dict(GATES.os.environ, {"VERUS": "/opt/verus"}, clear=True),
            mock.patch.object(GATES.subprocess, "run", side_effect=run) as process,
            mock.patch("builtins.print"),
        ):
            status = GATES.main()
        manifest, = (crate / "target/release-gates").glob("*/results.json")
        return status, json.loads(manifest.read_text()), manifest.parent, process.call_count

    def test_verus_gate_records_matching_counts_and_input_hashes(self):
        """A run that reproduces the README table passes and records its evidence."""

        counts = {"alpha.rs": 1200, "beta.rs": 677}
        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            verus_crate(crate, counts)
            status, result, output, calls = self.run_verus_fixture(crate, fake_verus(counts))
            self.assertEqual((status, calls), (0, 3))
            self.assertTrue(result["passed"])
            version, alpha, beta = result["gates"]
            self.assertEqual(version["verus_version"], GATES.VERUS_VERSION)
            self.assertNotIn("storage_preflight", version)
            self.assertEqual((alpha["harness"], alpha["verified"]), ("alpha.rs", 1200))
            self.assertEqual(beta["verus_evidence"], "verus.json")
            evidence = json.loads((output / "verus.json").read_text())
            self.assertEqual(evidence["harnesses"], counts)
            self.assertEqual((evidence["verified"], evidence["errors"]), (1877, 0))
            self.assertIn("--no-cheating", evidence["flags"])
            self.assertEqual(set(evidence["inputs"]["alpha.rs"]["sources"]), {"src/alpha.rs"})
            self.assertEqual(len(evidence["inputs"]["beta.rs"]["harness"]["sha256"]), 64)

    def test_verus_gate_rejects_counts_that_drift_from_the_readme(self):
        """A proof change without a table update fails and names the harness to update."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            verus_crate(crate, {"alpha.rs": 10, "beta.rs": 5})
            status, result, output, calls = self.run_verus_fixture(crate, fake_verus({"alpha.rs": 11, "beta.rs": 5}))
            self.assertEqual((status, calls), (1, 2))
            self.assertFalse(result["passed"])
            row = result["gates"][-1]
            self.assertEqual(row["process_exit_code"], 0)
            self.assertIn("alpha.rs: Verus verified 11 items, but verus/README.md records 10", row["validation_error"])
            self.assertFalse((output / "verus.json").exists())

    def test_verus_gate_rejects_a_wrong_prover_or_a_failed_proof(self):
        """A different release or a harness with errors never reaches the evidence file."""

        counts = {"alpha.rs": 10}
        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            verus_crate(crate, counts)

            def other_release(command, **kwargs):
                """Report a different Verus release."""

                kwargs["stdout"].write(VERUS_VERSION_OUTPUT.replace("426d8b0", "0000000"))
                return subprocess.CompletedProcess(command, 0)

            status, result, _, calls = self.run_verus_fixture(crate, other_release)
            self.assertEqual((status, calls), (1, 1))
            self.assertIn("pinned", result["gates"][0]["validation_error"])
        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            verus_crate(crate, counts)

            def failing(command, **kwargs):
                """Fail the proof with Verus's nonzero exit status."""

                if command[1:] == ["--version"]:
                    kwargs["stdout"].write(VERUS_VERSION_OUTPUT)
                    return subprocess.CompletedProcess(command, 0)
                kwargs["stdout"].write("verification results:: 9 verified, 1 errors\n")
                return subprocess.CompletedProcess(command, 1)

            status, result, output, _ = self.run_verus_fixture(crate, failing)
            self.assertEqual(status, 1)
            self.assertEqual(result["gates"][-1]["exit_code"], 1)
            self.assertFalse((output / "verus.json").exists())

    def test_verus_gate_rejects_inputs_edited_during_the_run(self):
        """Counts are bound to the exact harness and source bytes the prover read."""

        counts = {"alpha.rs": 10, "beta.rs": 5}
        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            verus_crate(crate, counts)

            def edit(harness):
                """Change an included source after its harness was verified."""

                if harness.name == "beta.rs":
                    (crate / "src/alpha.rs").write_text("pub fn alpha() -> u8 { 2 }\n")

            status, result, output, _ = self.run_verus_fixture(crate, fake_verus(counts, edit))
            self.assertEqual(status, 1)
            self.assertIn("changed while the gate ran", result["gates"][-1]["validation_error"])
            self.assertFalse((output / "verus.json").exists())

    def test_verus_gate_rejects_a_table_that_omits_a_harness(self):
        """A harness missing from the README table fails before the prover starts."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            verus_crate(crate, {"alpha.rs": 10})
            (crate / "src/gamma.rs").write_text("pub fn gamma() {}\n")
            (crate / "verus/gamma.rs").write_text('//! Harness.\n#[path = "../src/gamma.rs"]\nmod gamma;\n')
            status, result, _, calls = self.run_verus_fixture(crate, fake_verus({"alpha.rs": 10, "gamma.rs": 1}))
            self.assertEqual((status, calls), (1, 0))
            self.assertIn("gamma.rs", result["gates"][0]["validation_error"])

    def test_package_listing_checks_the_actual_cargo_inventory(self):
        """Allow model source but reject artifact trees, weights, and bytecode."""

        base = "Cargo.toml\nCargo.lock\nREADME.md\nsrc/lib.rs\n"
        self.assertEqual(GATES.validate_package_listing(base + "src/models/t5.rs\n"), 5)
        self.assertEqual(GATES.validate_package_listing(base + "scripts/COVERAGE_EVIDENCE.md\n"), 5)
        self.assertEqual(GATES.validate_package_listing(base + "scripts/WHOLE_FILE_COVERAGE.md\n"), 5)
        self.assertEqual(GATES.validate_package_listing("warning: manifest has no homepage\n  |\n  = note: docs\n" + base), 4)
        for entry in ("models/model/README.md", "target/package.md", "fuzz/README.md", "nested/Cargo.lock", "tests/weights.safetensors", "tests/weights.bpk", "scripts/__pycache__/x.pyc", "scripts/coverage-layout/target/debug/build.rs", "scripts/coverage-layout/Cargo.toml", "scripts/coverage-layout/src/lib.rs", "scripts/.git/config", "/tmp/README.md", "src/../secret.md", "Cargo.lock"):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                GATES.validate_package_listing(base + entry + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            GATES.validate_package_listing("Cargo.toml\n")

    def test_package_listing_accepts_only_small_tiny_golden_fixtures(self):
        """Tiny random-weight goldens may ship; other or oversized weights may not."""

        base = "Cargo.toml\nCargo.lock\nREADME.md\nsrc/lib.rs\n"
        fixture = "tests/fixtures/tiny_goldens/llama3/model.safetensors"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / fixture).parent.mkdir(parents=True)
            (root / fixture).write_bytes(b"\0" * 1024)
            self.assertEqual(GATES.validate_package_listing(base + fixture + "\n", root), 5)
            for entry in ("tests/fixtures/tiny_goldens/qwen2/model.safetensors",  # absent
                          "tests/fixtures/model.safetensors",
                          "tests/fixtures/tiny_goldens/llama3/nested/model.safetensors",
                          "tests/fixtures/tiny_goldens/llama3/model.bin",
                          "tests/fixtures/tiny_goldens/llama3/model.bpk"):
                (root / entry).parent.mkdir(parents=True, exist_ok=True)
                if "qwen2" not in entry:
                    (root / entry).write_bytes(b"\0" * 1024)
                with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, "runtime artifact"):
                    GATES.validate_package_listing(base + entry + "\n", root)
            large = "tests/fixtures/tiny_goldens/gpt-oss/model.safetensors"
            (root / large).parent.mkdir(parents=True)
            with (root / large).open("wb") as handle:
                handle.truncate(GATES.TINY_GOLDEN_MAX_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "runtime artifact"):
                GATES.validate_package_listing(base + large + "\n", root)
            linked = "tests/fixtures/tiny_goldens/olmo2/model.safetensors"
            (root / linked).parent.mkdir(parents=True)
            (root / linked).symlink_to(root / fixture)
            with self.assertRaisesRegex(ValueError, "runtime artifact"):
                GATES.validate_package_listing(base + linked + "\n", root)

    def test_package_listing_accepts_the_repository_tiny_goldens(self):
        """The committed goldens that the packaged tests read pass the listing check."""

        fixtures = sorted(path.relative_to(GATES.CRATE).as_posix()
                          for path in (GATES.CRATE / GATES.TINY_GOLDEN_FIXTURES).glob("*/*.safetensors"))
        self.assertTrue(fixtures)
        listing = "Cargo.toml\nCargo.lock\nREADME.md\nsrc/lib.rs\n" + "".join(f"{entry}\n" for entry in fixtures)
        self.assertEqual(GATES.validate_package_listing(listing), 4 + len(fixtures))


if __name__ == "__main__":
    unittest.main()
