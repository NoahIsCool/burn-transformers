"""Whole-file joins must retain production/ambiguous source and reject stale evidence."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "coverage_exclusions.py"
SPEC = importlib.util.spec_from_file_location("coverage_exclusions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
E = importlib.util.module_from_spec(SPEC)
with mock.patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    SPEC.loader.exec_module(E)
    import coverage_binding as B
    import check_release
    from coverage_provenance import capture_sources, fingerprint_file, require_same_sources


def put(path, text):
    """Write only fixture-owned files and return their exact identities."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return fingerprint_file(path)


def file_report(path, identity, test_only):
    """Represent a complete synthetic whole-file context, not real compiler proof."""

    start = {"byte": 0, "line": 1, "column": 1, "unicode_column": 1}
    end = {"byte": identity["bytes"], "line": identity["lines"] + 1, "column": 1, "unicode_column": 1}
    return {"path": path, "sha256": identity["sha256"], "byte_length": identity["bytes"],
            "line_count": identity["lines"], "contexts": [{"module": "entry[0]::fixture", "duplicate_context": False,
                "regions": [{"span": {"start": start, "end": end}, "depth": 0, "kind": "file",
                             "test_only": test_only, "reason": "synthetic fixture"}]}],
            "lines": [] if not identity["lines"] else [{"start_line": 1, "end_line": identity["lines"],
                        "classification": "test_only" if test_only else "retained"}]}


def fixture(crate):
    """Build source identities, synthetic contexts and a segment-level export."""

    put(crate / "Cargo.toml", '[package]\nname="fixture"\nversion="0.1.0"\n[workspace]\n')
    put(crate / "Cargo.lock", 'version=4\n')
    sources = {"src/lib.rs": "pub fn value() -> u32 { 7 }\n", "src/unit.rs": "fn unit() {}\n",
               "src/models/experimental.rs": "fn experimental() {}\n", "src/tests/production.rs": "fn live() {}\n",
               "tests/integration.rs": "fn integration() {}\n"}
    for name, text in sources.items():
        put(crate / name, text)
    snapshot = capture_sources(crate)
    production_inputs = sorted(["src/lib.rs", "src/tests/production.rs", "src/models/experimental.rs"])
    library = {"name": "fixture", "source": str(crate / "src/lib.rs"), "test": False}
    targets = [{"target": library, "inputs": production_inputs},
               {"target": {**library, "test": True}, "inputs": sorted([*production_inputs, "src/unit.rs"])},
               {"target": {"name": "integration", "source": str(crate / "tests/integration.rs"), "test": True},
                "inputs": ["tests/integration.rs"]}]
    classification = {"schema_version": 1, "purpose": "supplementary_source_classification_not_a_coverage_gate",
                      "qualified_for_coverage_exclusions": False, "macro_expansion_resolved": False,
                      "root": str(crate), "entries": E.target_entries(targets, crate), "diagnostics": [],
                      "files": [file_report(name, snapshot["files"][name], name == "src/unit.rs") for name in sources]}
    native = {"type": "llvm.coverage.json.export", "version": "3.0.1", "data": [{"files": [
        {"filename": str(crate / name), "segments": [[1, 1, int(name != "src/models/experimental.rs"), True, True, False],
                                                    [1, len(text), 0, False, False, False]],
         "summary": {"lines": {"count": 2, "covered": 0 if name == "src/models/experimental.rs" else 2}}}
        for name, text in sources.items()]}]}
    return snapshot, classification, targets, native


class WholeFileTests(unittest.TestCase):
    """Pure adversarial fixtures never authorize real artifact qualification."""

    def setUp(self):
        """Prepare one isolated source tree for each independent contract test."""

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.crate = Path(self.temporary.name).resolve()
        self.snapshot, self.classification, self.targets, self.native = fixture(self.crate)

    def diagnostic(self):
        """Apply the pure helper to this test's explicit synthetic observations."""

        return E.whole_file_diagnostic(self.snapshot, self.classification, self.targets, self.native)

    def unit(self):
        """Find the one wholly test-owned source candidate."""

        return next(file for file in self.classification["files"] if file["path"] == "src/unit.rs")

    def assert_retained_unit(self):
        """Require ambiguity or production references to preserve all unit-file lines."""

        row = next(row for row in self.diagnostic()["files"] if row["path"] == "src/unit.rs")
        self.assertFalse(row["excluded"])
        self.assertEqual(row["removed_mapped_lines"], 0)

    def test_only_whole_test_candidate_is_removed_and_counters_stay_distinct(self):
        """Neither test directory names nor experimental-family status is a waiver."""

        result = self.diagnostic()
        self.assertEqual([row["path"] for row in result["files"] if row["excluded"]], ["src/unit.rs"])
        self.assertEqual(result["legacy_native_aggregate"]["crate"]["lines"], 8)
        self.assertEqual(result["physical_before_exclusions"]["crate"]["lines"], 4)
        self.assertEqual(result["physical_whole_file_diagnostic"]["crate"]["lines"], 3)
        self.assertEqual(result["physical_whole_file_diagnostic"]["shared"]["lines"], 2)
        for field in ("production_only_exclusions_qualified", "production_only_measurement",
                      "compiler_dependency_closure_verified", "release_gate_promotion"):
            self.assertFalse(result[field])
        self.assertNotIn("whole_test_file_exclusions_qualified", result)
        self.assertEqual(next(row for row in result["files"] if row["excluded"])["removed_covered_lines"], 1)

    def test_production_input_vetoes_even_a_syntactic_test_candidate(self):
        """Compiler-discovered macro/include edges override the parsed module graph."""

        self.targets[0]["inputs"].append("src/unit.rs")
        self.targets[0]["inputs"].sort()
        self.assert_retained_unit()

    def test_missing_unit_observation_and_classifier_file_are_retained(self):
        """Absence from production alone is insufficient evidence for removing source."""

        self.targets[1]["inputs"].remove("src/unit.rs")
        self.targets[2]["inputs"].insert(0, "src/unit.rs")
        self.assert_retained_unit()
        self.classification["files"].remove(self.unit())
        self.assert_retained_unit()

    def test_production_alias_mixed_line_and_unknown_diagnostics_are_retained(self):
        """Any unknown context, even on blank bytes, vetoes a whole-file exclusion."""

        baseline = copy.deepcopy(self.classification)
        for case in ("context", "mixed", "retained", "diagnostic", "opaque", "inner_region"):
            with self.subTest(case=case):
                self.classification = copy.deepcopy(baseline)
                unit = self.unit()
                if case == "context":
                    context = copy.deepcopy(unit["contexts"][0])
                    context["regions"][0]["test_only"] = False
                    unit["contexts"].append(context)
                elif case in ("mixed", "retained"):
                    unit["lines"][0]["classification"] = case
                elif case in ("diagnostic", "opaque"):
                    self.classification["diagnostics"].append({"file": "src/unit.rs", "code": "ambiguous_path" if case == "diagnostic" else "opaque_macro_or_verbatim"})
                else:
                    region = copy.deepcopy(unit["contexts"][0]["regions"][0])
                    region.update(depth=1, kind="item", test_only=False)
                    unit["contexts"][0]["regions"].append(region)
                self.assert_retained_unit()

    def test_equivalent_duplicate_context_is_not_confused_with_ambiguity(self):
        """Repeated positive observations remain represented but do not forbid exclusion."""

        context = copy.deepcopy(self.unit()["contexts"][0])
        context["duplicate_context"] = True
        self.unit()["contexts"].append(context)
        self.classification["diagnostics"].append({"file": "src/unit.rs", "code": "duplicate_context"})
        self.assertTrue(next(row for row in self.diagnostic()["files"] if row["path"] == "src/unit.rs")["excluded"])

    def test_malformed_line_and_context_contracts_fail_instead_of_becoming_empty(self):
        """Missing, overlapping, truncated, nonintegral and contradictory structures fail."""

        baseline = copy.deepcopy(self.classification)
        cases = ("no_context", "no_regions", "wrong_root", "root_depth", "root_span", "duplicate_root",
                 "no_lines", "line_gap", "line_overlap", "line_bool", "unknown_class", "test_bool", "bad_span", "reverse_span")
        for case in cases:
            with self.subTest(case=case):
                self.classification = copy.deepcopy(baseline)
                unit = self.unit()
                root = unit["contexts"][0]["regions"][0]
                if case == "no_context": unit["contexts"] = []
                elif case == "no_regions": unit["contexts"][0]["regions"] = []
                elif case == "wrong_root": root["kind"] = "item"
                elif case == "root_depth": root["depth"] = 1
                elif case == "root_span": root["span"]["start"]["byte"] = 1
                elif case == "duplicate_root": unit["contexts"][0]["regions"].append(copy.deepcopy(root))
                elif case == "no_lines": unit["lines"] = []
                elif case == "line_gap": unit["lines"][0]["start_line"] = 2
                elif case == "line_overlap": unit["lines"].append(copy.deepcopy(unit["lines"][0]))
                elif case == "line_bool": unit["lines"][0]["end_line"] = True
                elif case == "unknown_class": unit["lines"][0]["classification"] = "generated"
                elif case == "test_bool": root["test_only"] = 1
                elif case == "bad_span": root["span"]["end"]["line"] = 500
                elif case == "reverse_span": root["span"]["start"]["byte"], root["span"]["end"]["byte"] = 5, 1
                with self.assertRaises(ValueError):
                    self.diagnostic()

    def test_identity_entries_target_and_input_mismatches_fail_closed(self):
        """A same-named source or false qualification bit cannot replace evidence."""

        baseline_classification, baseline_targets = copy.deepcopy(self.classification), copy.deepcopy(self.targets)
        cases = ("source_hash", "entry_missing", "entry_extra", "production_missing", "extra_production", "unit_missing",
                 "duplicate_target", "target_alias", "missing_input", "duplicate_input", "input_traversal", "missing_root")
        for case in cases:
            with self.subTest(case=case):
                self.classification, self.targets = copy.deepcopy(baseline_classification), copy.deepcopy(baseline_targets)
                if case == "source_hash": self.unit()["sha256"] = "0" * 64
                elif case == "entry_missing": self.classification["entries"].pop()
                elif case == "entry_extra": self.classification["entries"].append("src/unit.rs")
                elif case == "production_missing": self.targets[0]["target"]["test"] = True
                elif case == "extra_production": self.targets[2]["target"]["test"] = False
                elif case == "unit_missing": self.targets[1]["target"]["name"] = "other"
                elif case == "duplicate_target": self.targets.append(copy.deepcopy(self.targets[0]))
                elif case == "target_alias": self.targets[0]["target"]["source"] = str(self.crate) + "/src/../src/lib.rs"
                elif case == "missing_input": self.targets[1]["inputs"] = []
                elif case == "duplicate_input": self.targets[1]["inputs"] *= 2
                elif case == "input_traversal": self.targets[1]["inputs"] = ["src/../src/lib.rs"]
                elif case == "missing_root": self.targets[1]["inputs"].remove("src/lib.rs")
                with self.assertRaises(ValueError): self.diagnostic()

    def test_native_aliases_bounds_unbound_files_and_invalid_counters_fail(self):
        """A denominator cannot double-count aliases or trust out-of-source line mappings."""

        baseline = copy.deepcopy(self.native)
        for case in ("alias", "unknown_source", "non_rust", "line_bounds", "legacy_bool", "legacy_covered", "missing_segments", "duplicate_file"):
            with self.subTest(case=case):
                self.native = copy.deepcopy(baseline)
                files = self.native["data"][0]["files"]
                if case == "alias":
                    extra = copy.deepcopy(files[0]); extra["filename"] = "src/../src/lib.rs"; files.append(extra)
                elif case == "unknown_source": files[0]["filename"] = str(self.crate / "src/unknown.rs")
                elif case == "non_rust": files[0]["filename"] = str(self.crate / "src/config.json")
                elif case == "line_bounds": files[0]["segments"] = [[2, 1, 1, True, True, False], [2, 9, 0, False, False, False]]
                elif case == "legacy_bool": files[0]["summary"]["lines"]["count"] = True
                elif case == "legacy_covered": files[0]["summary"]["lines"]["covered"] = 3
                elif case == "missing_segments": del files[0]["segments"]
                elif case == "duplicate_file": files.append(copy.deepcopy(files[0]))
                with self.assertRaises(ValueError): self.diagnostic()

    def test_integer_threshold_boundaries_remain_90_and_95(self):
        """No rounding or coverage-exclusion policy changes either required threshold."""

        for covered, expected in ((94, False), (95, True), (96, True)):
            result = E._thresholds([{"path": "src/core.rs", "n": 100, "c": covered}], "n", "c")
            self.assertEqual(result["shared"]["meets_threshold"], expected)
        for covered, expected in ((89, False), (90, True), (91, True)):
            result = E._thresholds([{"path": "src/core.rs", "n": 100, "c": covered}], "n", "c")
            self.assertEqual(result["crate"]["meets_threshold"], expected)
        with self.assertRaises(ValueError): E._thresholds([], "n", "c")


class ClassifierCaptureTests(unittest.TestCase):
    """Run-local invocation identity is independently necessary for qualification."""

    def setUp(self):
        """Reuse source construction, without repeating the pure policy test suite."""

        WholeFileTests.setUp(self)

    def capture(self, *, mode="success"):
        """Simulate native outputs explicitly; this helper is not a compiler fixture."""

        executable = self.crate / "target/classifier"
        put(executable, "#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)
        output = self.crate / "target/classification"

        def run(command, **kwargs):
            """Populate literal subprocess streams and optionally change their inputs."""

            kwargs["stdout"].write(json.dumps(self.classification).encode())
            if mode == "diagnostic": kwargs["stderr"].write(b"warning: classifier fixture\n")
            if mode == "source_change": put(self.crate / "src/unit.rs", "fn changed() {}\n")
            if mode == "tool_change": put(executable, "#!/bin/sh\nexit 1\n")
            if mode == "timeout": raise subprocess.TimeoutExpired(command, 1)
            return subprocess.CompletedProcess(command, 7 if mode == "native_failure" else 0)

        with mock.patch.object(E.subprocess, "run", side_effect=run):
            return E.capture_classifier(self.crate, self.snapshot, executable, output, self.classification["entries"])

    def test_fresh_capture_binds_invocation_output_tool_and_scope(self):
        """Original stdout bytes and each explicit root remain attached to the call."""

        record = self.capture()
        path = self.crate / "target/classification/invocation.json"
        self.assertTrue(record["capture_valid"])
        self.assertEqual(E.validated_classifier(path, self.snapshot, self.classification["entries"]), self.classification)
        with self.assertRaises(FileExistsError): self.capture()

    def test_nonzero_diagnostics_mutations_and_timeouts_retain_failed_record(self):
        """A failed invocation never turns into an empty or qualified classification."""

        for mode in ("native_failure", "diagnostic", "source_change", "tool_change", "timeout"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                original = self.crate, self.snapshot, self.classification, self.targets, self.native
                self.crate = Path(directory).resolve()
                self.snapshot, self.classification, self.targets, self.native = fixture(self.crate)
                try:
                    with self.assertRaises((ValueError, subprocess.TimeoutExpired)): self.capture(mode=mode)
                    record = E.read_json(self.crate / "target/classification/invocation.json")
                    self.assertFalse(record["capture_valid"])
                    self.assertIn("capture_error", record)
                finally:
                    self.crate, self.snapshot, self.classification, self.targets, self.native = original

    def test_retained_invocation_mutations_are_rejected(self):
        """No stale log/report, changed executable or forged target list is adopted."""

        self.capture()
        path = self.crate / "target/classification/invocation.json"
        original = path.read_text()
        for case in ("exit_bool", "capture_valid", "cwd", "entries", "command", "inventory", "report_path", "stderr_nonempty"):
            with self.subTest(case=case):
                record = json.loads(original)
                if case == "exit_bool": record["process_exit_code"] = False
                elif case == "capture_valid": record["capture_valid"] = False
                elif case == "cwd": record["cwd"] = "/other"
                elif case == "entries": record["entries"] = []
                elif case == "command": record["command"].append("src/unit.rs")
                elif case == "inventory": record["source_inventory_sha256"] = "0" * 64
                elif case == "report_path": record["stdout"]["path"] = str(self.crate / "elsewhere.json")
                elif case == "stderr_nonempty": record["stderr"]["identity"]["bytes"] = 1
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError): E.validated_classifier(path, self.snapshot, self.classification["entries"])
        path.write_text(original)
        put(self.crate / "target/classification/classification.json", "{}")
        with self.assertRaises(ValueError): E.validated_classifier(path, self.snapshot, self.classification["entries"])


@unittest.skipUnless(os.environ.get("TFM_WHOLE_FILE_COVERAGE_NATIVE_TESTS") == "1", "opt-in tiny pinned compiler/classifier fixture")
class NativeWholeFileTests(unittest.TestCase):
    """Native macro/include/configuration inputs must veto syntactic candidates."""

    def test_native_compiler_edges_override_the_classifier_without_inline_exclusions(self):
        """Build only a tiny offline fixture and the independent development classifier."""

        with tempfile.TemporaryDirectory(prefix="tfm-whole-file-native-") as directory:
            root = Path(directory).resolve()
            environment = {key: value for key, value in os.environ.items() if key not in (
                "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
                "CARGO_TARGET_DIR", "CARGO_BUILD_TARGET", "LLVM_COV", "LLVM_PROFDATA", "LLVM_PROFILE_FILE")}
            environment.update(CARGO_INCREMENTAL="0", CARGO_TERM_COLOR="never")
            version = subprocess.check_output(["rustc", "+nightly", "-Vv"], env=environment, text=True, timeout=30)
            B.producer_from_version(version)
            classifier_target = root / "classifier target with spaces"
            manifest = SCRIPT.parent / "coverage-layout/Cargo.toml"
            classifier_sources = {str(path): fingerprint_file(path) for path in [manifest, manifest.with_name("Cargo.lock"),
                *sorted(manifest.parent.joinpath("src").glob("*.rs"))]}
            build = subprocess.run(["cargo", "+nightly", "build", "--manifest-path", str(manifest), "--locked", "--offline",
                                    "--target-dir", str(classifier_target)], env=environment, capture_output=True, text=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            self.assertNotIn("warning:", build.stderr)
            for path, identity in classifier_sources.items(): self.assertEqual(fingerprint_file(Path(path)), identity)
            executable = classifier_target / "debug/transformers-coverage-layout"
            crate = root / "fixture with spaces"
            put(crate / "Cargo.toml", '[package]\nname="whole_file_fixture"\nversion="0.1.0"\nedition="2024"\n[workspace]\n[features]\nfixture=[]\nunselected=[]\n')
            put(crate / "Cargo.lock", 'version=4\n[[package]]\nname="whole_file_fixture"\nversion="0.1.0"\n')
            put(crate / "src/lib.rs", '''#![allow(dead_code)]
#[path = "tests/production.rs"] mod production;
#[cfg(test)] #[path = "unit only.rs"] mod unit_only;
#[cfg(test)] #[path = "aliased.rs"] mod test_alias;
#[path = "aliased.rs"] mod production_alias;
#[cfg(test)] #[path = "macro_module.rs"] mod test_macro;
macro_rules! production_module { () => { #[path = "macro_module.rs"] mod macro_live; }; }
production_module!();
#[cfg(test)] #[path = "included.rs"] mod test_include;
mod included { include!("included.rs"); }
#[cfg(test)] #[path = "raw_input.rs"] mod test_raw;
const RAW: &str = include_str!("raw_input.rs");
#[cfg(all(test, feature = "fixture"))] #[path = "feature_test.rs"] mod feature_test;
#[cfg(any(test, feature = "unselected"))] mod uncertain;
#[cfg(test)] #[cfg_attr(feature = "fixture", path = "conditional.rs")] mod conditional;
pub fn value() -> u32 { production::value() + production_alias::value() + macro_live::value() + included::value() }
''')
            for name in ("tests/production.rs", "aliased.rs", "macro_module.rs", "included.rs", "raw_input.rs", "uncertain.rs", "conditional.rs"):
                put(crate / "src" / name, 'pub fn value() -> u32 { 1 }\n#[test] fn own_test() { assert_eq!(value(), 1); }\n')
            for name in ("unit only.rs", "feature_test.rs"):
                put(crate / "src" / name, '#[test] fn check() { assert_eq!(super::value(), 4); }\n')
            put(crate / "tests/integration.rs", '#[test] fn check() { assert_eq!(whole_file_fixture::value(), 4); }\n')
            output = crate / "target/evidence"
            output.mkdir(parents=True)
            configured, setup = B.prepare_capture(crate, output, "nightly", "fixture", environment)
            before = capture_sources(crate)
            log = output / "coverage.log"
            command = ["cargo", "+nightly", "llvm-cov", "--manifest-path", str(crate / "Cargo.toml"), "--locked", "--offline",
                       "--no-default-features", "--features", "fixture", "--lib", "--tests", "--json",
                       "--output-path", str(output / "coverage.json"), "--disable-default-ignore-filename-regex"]
            with log.open("xb") as stream:
                run = subprocess.run(command, cwd=crate, env=configured, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(run.returncode, 0, log.read_text())
            require_same_sources(before, capture_sources(crate))
            binding = B.validate_capture(setup, before, crate)
            capture_dir = output / "classifier"
            E.capture_classifier(crate, before, executable, capture_dir, E.target_entries(binding["targets"], crate))
            result = E.qualify_capture(setup, before, capture_dir / "invocation.json", crate, coverage_log=log)
            rows = {row["path"]: row for row in result["files"]}
            self.assertEqual({name for name, row in rows.items() if row["excluded"]}, {"src/unit only.rs", "src/feature_test.rs"})
            for name in ("tests/production.rs", "aliased.rs", "macro_module.rs", "included.rs", "raw_input.rs"):
                self.assertEqual(rows["src/" + name]["reason"], "retained_production_compiler_input")
            for name in ("uncertain.rs", "conditional.rs"):
                self.assertEqual(rows["src/" + name]["reason"], "retained_unknown_mixed_or_non_test_file_ownership")
            self.assertTrue(result["whole_test_file_exclusions_qualified"])
            self.assertFalse(result["production_only_exclusions_qualified"])
            self.assertFalse(result["release_gate_promotion"])
            with log.open("a") as stream: stream.write("warning: mismatched profile fixture\n")
            with self.assertRaisesRegex(ValueError, "unqualified coverage diagnostic"):
                E.qualify_capture(setup, before, capture_dir / "invocation.json", crate, coverage_log=log)


if __name__ == "__main__":
    unittest.main()
