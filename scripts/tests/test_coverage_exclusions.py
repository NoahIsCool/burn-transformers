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

    def add_unit_macro(self):
        """Attach one exact synthetic macro span without claiming expanded syntax."""

        region = copy.deepcopy(self.unit()["contexts"][0]["regions"][0])
        region.update(depth=1, kind="macro")
        self.unit()["contexts"][0]["regions"].append(region)
        self.classification["diagnostics"].append({"file": "src/unit.rs", "code": "opaque_macro_or_verbatim",
                                                  "span": copy.deepcopy(region["span"])})
        return region

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

    def test_exact_macros_in_every_whole_test_context_do_not_infer_expansion_ownership(self):
        """Whole-file proof, not a property-library name or macro expansion, qualifies."""

        self.add_unit_macro()
        context = copy.deepcopy(self.unit()["contexts"][0])
        context.update(module="entry[0]::second_test_alias", duplicate_context=True)
        self.unit()["contexts"].append(context)
        self.classification["diagnostics"].append({"file": "src/unit.rs", "code": "duplicate_context"})
        result = self.diagnostic()
        self.assertEqual([row["path"] for row in result["files"] if row["excluded"]], ["src/unit.rs"])
        self.assertFalse(result["compiler_dependency_closure_verified"])
        self.assertFalse(result["production_only_measurement"])
        self.assertEqual(result["policy"], "whole_test_files_v2_exact_test_macro_spans_no_inline_or_generated_exclusions")

    def test_macro_diagnostics_cannot_waive_production_cfg_path_or_context_vetoes(self):
        """Matching an opaque span never overrides any independent retention reason."""

        self.add_unit_macro()
        baseline, targets = copy.deepcopy(self.classification), copy.deepcopy(self.targets)
        for case in ("production_input", "production_context", "inline_root", "mixed_line", "unknown_cfg",
                     "ambiguous_path", "missing_in_second_context", "different_kind_in_second_context"):
            with self.subTest(case=case):
                self.classification, self.targets = copy.deepcopy(baseline), copy.deepcopy(targets)
                contexts = self.unit()["contexts"]
                if case == "production_input":
                    self.targets[0]["inputs"] = sorted([*self.targets[0]["inputs"], "src/unit.rs"])
                elif case == "production_context":
                    context = copy.deepcopy(contexts[0]); context["regions"][0]["test_only"] = False
                    contexts.append(context)
                elif case == "inline_root": contexts[0]["regions"][0]["test_only"] = False
                elif case == "mixed_line": self.unit()["lines"][0]["classification"] = "mixed"
                elif case in ("unknown_cfg", "ambiguous_path"):
                    self.classification["diagnostics"].append({"file": "src/unit.rs", "code": case})
                else:
                    context = copy.deepcopy(contexts[0])
                    if case == "missing_in_second_context": context["regions"].pop()
                    else: context["regions"][-1]["kind"] = "verbatim"
                    contexts.append(context)
                self.assert_retained_unit()

    def test_macro_diagnostic_requires_exact_nonempty_parsed_macro_span(self):
        """Verbatim, nearby regions, missing positions and partial matches stay counted."""

        self.add_unit_macro()
        baseline = copy.deepcopy(self.classification)
        for case in ("missing", "null", "byte", "line", "column", "unicode_column", "verbatim",
                     "unresolved_item", "impl_item", "not_test_only", "empty", "extra_unmatched"):
            with self.subTest(case=case):
                self.classification = copy.deepcopy(baseline)
                region = self.unit()["contexts"][0]["regions"][-1]
                issue = self.classification["diagnostics"][0]
                if case == "missing": issue.pop("span")
                elif case == "null": issue["span"] = None
                elif case in ("byte", "line", "column", "unicode_column"): issue["span"]["start"][case] += 1
                elif case in ("verbatim", "unresolved_item", "impl_item"): region["kind"] = case
                elif case == "not_test_only": region["test_only"] = False
                elif case == "empty":
                    region["span"]["end"] = copy.deepcopy(region["span"]["start"])
                    issue["span"] = copy.deepcopy(region["span"])
                else:
                    issue = copy.deepcopy(issue); issue["span"]["start"]["byte"] += 1
                    self.classification["diagnostics"].append(issue)
                self.assert_retained_unit()

    def test_malformed_macro_coordinates_fail_instead_of_authorizing_exclusion(self):
        """Malformed opaque-diagnostic evidence must not become a benign exception."""

        self.add_unit_macro()
        baseline = copy.deepcopy(self.classification)
        for case in ("not_object", "missing_coordinate", "boolean", "out_of_bounds", "reversed"):
            with self.subTest(case=case):
                self.classification = copy.deepcopy(baseline)
                issue = self.classification["diagnostics"][0]
                if case == "not_object": issue["span"] = []
                elif case == "missing_coordinate": del issue["span"]["start"]["unicode_column"]
                elif case == "boolean": issue["span"]["start"]["byte"] = False
                elif case == "out_of_bounds": issue["span"]["end"]["line"] = 500
                else: issue["span"]["start"]["byte"], issue["span"]["end"]["byte"] = 5, 1
                with self.assertRaises(ValueError): self.diagnostic()

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

    def retained_run(self):
        """Construct synthetic retained-row contracts, never native artifact proof."""

        root = self.crate / "target/run"
        setup = {"compiler_root": str(root / "compiler"), "export_root": str(root / "llvm-export"),
                 "artifact_root": str(self.crate / "target/llvm-cov-target")}
        binding = {"artifact_binding_verified": True, "native_report": {"fixture": "separately revalidated"}}
        command = ["cargo", "+nightly", "llvm-cov", "--disable-default-ignore-filename-regex"]
        before = {"sources": self.snapshot, "working_directory": str(self.crate), "command": command,
                  "runner_inputs": {}, "tool_versions": [], "compiler_environment": {
                      "TFM_COVERAGE_CAPTURE_ROOT": setup["compiler_root"],
                      "TFM_COVERAGE_ARTIFACT_ROOT": setup["artifact_root"],
                      "RUSTC_WORKSPACE_WRAPPER": str(root / "rustc-workspace-wrapper")}}
        for name, value in (("compiler-capture-setup.json", setup), ("compiler-binding.json", binding),
                            ("coverage-inputs-before.json", before), ("coverage-inputs-after.json", before)):
            put(root / name, json.dumps(value))
        log = root / "01-coverage.log"
        log_identity = put(log, check_release.UNFILTERED_OPTION_NOTICE + "\n")
        report_identity = put(root / "coverage.json", json.dumps(self.native))
        row = {"gate": "coverage", "command": command, "exit_code": 0, "process_exit_code": 0,
               "log": log.name, "log_identity": log_identity, "report_identity": report_identity,
               "compiler_evidence": {"setup": "compiler-capture-setup.json", "binding": "compiler-binding.json",
                    "artifact_binding_verified": True, "compiler_reported_inputs_verified": True, "no_export_filter_verified": True},
               "input_evidence": {"before": "coverage-inputs-before.json", "after": "coverage-inputs-after.json", "sources_unchanged": True}}
        results_path = root / "results.json"
        put(results_path, json.dumps({"schema_version": 1, "gates": [{"gate": "source"}, row], "passed": True}))
        return setup, binding, results_path, log

    def test_run_row_binds_exact_log_identity_and_all_retained_scope_records(self):
        """Success identifies the row's log plus command/setup/input/binding evidence."""

        setup, binding, results, log = self.retained_run()
        identities = E._validated_run_results(results, log, setup, self.snapshot, binding, self.crate)
        self.assertEqual(identities[str(log)], fingerprint_file(log))
        self.assertEqual(len(identities), 7)
        self.assertIn(str(results), identities)

    def test_clean_sibling_cannot_replace_the_actual_coverage_log(self):
        """A warning-bearing original is never rescued by a different clean sibling."""

        setup, binding, results, log = self.retained_run()
        sibling = log.parent / "clean-sibling.log"
        put(sibling, log.read_text())
        with log.open("a") as stream:
            stream.write("warning: mismatched profile fixture\n")
        with self.assertRaisesRegex(ValueError, "actual retained command log"):
            E._validated_run_results(results, sibling, setup, self.snapshot, binding, self.crate)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            E._validated_run_results(results, log, setup, self.snapshot, binding, self.crate)

    def test_failed_native_validation_and_changed_retained_scope_are_rejected(self):
        """Native nonzero, bool statuses, invented success and stale records all fail."""

        for case in ("native_failure", "native_bool", "validation_failure", "validation_error", "duplicate_coverage",
                     "renamed_log", "changed_setup", "changed_binding", "changed_command", "changed_after",
                     "changed_environment", "changed_report", "different_results"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                original_crate, original_snapshot = self.crate, self.snapshot
                self.crate = Path(directory).resolve()
                self.snapshot, _, _, _ = fixture(self.crate)
                try:
                    setup, binding, results, log = self.retained_run()
                    value = E.read_json(results)
                    row = value["gates"][1]
                    if case == "native_failure": row["process_exit_code"] = 7
                    elif case == "native_bool": row["process_exit_code"] = False
                    elif case == "validation_failure": row["exit_code"] = 1
                    elif case == "validation_error": row["validation_error"] = "mismatched profile"
                    elif case == "duplicate_coverage": value["gates"].append(copy.deepcopy(row))
                    elif case == "renamed_log": row["log"] = "clean-sibling.log"
                    elif case == "changed_setup": put(results.parent / "compiler-capture-setup.json", "{}")
                    elif case == "changed_binding": put(results.parent / "compiler-binding.json", "{}")
                    elif case == "changed_command": row["command"].append("--unexpected")
                    elif case in ("changed_after", "changed_environment"):
                        path = results.parent / "coverage-inputs-after.json"
                        context = E.read_json(path)
                        context["working_directory"] = "/different"
                        if case == "changed_environment":
                            context["working_directory"] = str(self.crate)
                            context["compiler_environment"]["TFM_COVERAGE_CAPTURE_ROOT"] = "/other/compiler"
                            put(results.parent / "coverage-inputs-before.json", json.dumps(context))
                        put(path, json.dumps(context))
                    elif case == "changed_report": put(results.parent / "coverage.json", "{}")
                    put(results, json.dumps(value))
                    if case == "different_results":
                        replacement = results.parent / "other-results.json"
                        put(replacement, results.read_text())
                        results = replacement
                    with self.assertRaises(ValueError):
                        E._validated_run_results(results, log, setup, self.snapshot, binding, self.crate)
                finally:
                    self.crate, self.snapshot = original_crate, original_snapshot

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


class StructuralAuditTests(unittest.TestCase):
    """Pure structural eligibility never substitutes for authenticated capture."""

    def setUp(self):
        """Reuse synthetic inputs with explicitly versioned empty structural audits."""

        WholeFileTests.setUp(self)
        for entry in self.classification["files"]:
            entry["test_audit"] = {"schema_version": 1, "complete": True, "macro_expansion_resolved": False, "sites": []}

    def result(self):
        """Return both unchanged counters and the independent structural predicate."""

        return E.whole_file_diagnostic(self.snapshot, self.classification, self.targets, self.native)

    def production(self):
        """Select an ordinary retained production source in the synthetic graph."""

        return next(entry for entry in self.classification["files"] if entry["path"] == "src/lib.rs")

    def site(self, kind="item", disposition="test_body"):
        """Attach an exact synthetic structural owner without source heuristics."""

        span = copy.deepcopy(self.production()["contexts"][0]["regions"][0]["span"])
        site = {"span": span, "attribute_span": copy.deepcopy(span), "owner_kind": kind, "disposition": disposition}
        self.production()["test_audit"]["sites"].append(site)
        return site

    def test_structural_eligibility_is_independent_of_thresholds_and_artifact_proof(self):
        """A valid layout does not imply sufficient exercised production code."""

        result = self.result()
        self.assertTrue(result["structural_test_audit"]["structural_eligibility"])
        self.assertFalse(result["physical_whole_file_diagnostic"]["crate"]["meets_threshold"])
        self.assertFalse(result["production_only_measurement"])
        self.assertFalse(result["production_only_exclusions_qualified"])
        self.assertFalse(result["compiler_dependency_closure_verified"])
        self.assertFalse(result["release_gate_promotion"])

    def test_each_residual_test_owner_or_unknown_test_dependency_blocks_without_exclusion(self):
        """Every explicit residual body stays counted and records an actionable blocker."""

        for kind in ("file", "item", "inline_mod", "impl_item", "trait_item", "expr", "stmt"):
            for disposition in ("test_body", "unresolved_test_dependent"):
                with self.subTest(kind=kind, disposition=disposition):
                    self.production()["test_audit"]["sites"] = []
                    self.site(kind, disposition)
                    result = self.result()
                    self.assertFalse(result["structural_test_audit"]["structural_eligibility"])
                    self.assertEqual(result["structural_test_audit"]["blockers"][0]["code"], disposition)
                    self.assertEqual(result["physical_whole_file_diagnostic"]["crate"]["lines"], 3)

    def test_resolved_external_declaration_can_share_a_counted_production_line(self):
        """The entire physical line remains; only the bodyless declaration is harmless."""

        site = self.site("external_mod", "external_declaration")
        region = copy.deepcopy(self.production()["contexts"][0]["regions"][0])
        region.update(kind="mod", depth=1, test_only=True)
        self.production()["contexts"][0]["regions"].append(region)
        self.production()["lines"][0]["classification"] = "mixed"
        self.classification["diagnostics"].append({"file": "src/lib.rs", "code": "mixed_production_test_lines"})
        self.assertEqual(region["span"], site["span"])
        result = self.result()
        self.assertTrue(result["structural_test_audit"]["structural_eligibility"])
        self.assertEqual(result["physical_whole_file_diagnostic"]["crate"]["lines"], 3)
        self.classification["diagnostics"].append({"file": "src/lib.rs", "code": "ambiguous_module_path"})
        self.assertFalse(self.result()["structural_test_audit"]["structural_eligibility"])

    def test_old_incomplete_unit_only_unclassified_and_test_context_only_inputs_block(self):
        """Compiler input membership alone cannot turn included text into runtime code."""

        baseline, targets = copy.deepcopy(self.classification), copy.deepcopy(self.targets)
        for case in ("old", "incomplete", "version", "unit_only", "unclassified", "test_context"):
            with self.subTest(case=case):
                self.classification, self.targets = copy.deepcopy(baseline), copy.deepcopy(targets)
                if case == "old": self.production().pop("test_audit")
                elif case == "incomplete": self.production()["test_audit"]["complete"] = False
                elif case == "version": self.production()["test_audit"]["schema_version"] = 99
                elif case == "unit_only": self.targets[0]["inputs"].remove("src/models/experimental.rs")
                elif case == "unclassified": self.classification["files"].remove(self.production())
                else: self.production()["contexts"][0]["regions"][0]["test_only"] = True
                result = self.result()
                self.assertFalse(result["structural_test_audit"]["structural_eligibility"])
                self.assertTrue(result["structural_test_audit"]["blockers"])
                self.assertEqual(result["physical_whole_file_diagnostic"]["crate"]["lines"], 3)

    def test_unknown_runtime_macros_remain_counted_without_a_global_closure_claim(self):
        """An opaque runtime diagnostic is not an exclusion or expansion attestation."""

        self.classification["diagnostics"].append({"file": "src/lib.rs", "code": "opaque_macro_or_verbatim"})
        result = self.result()
        self.assertTrue(result["structural_test_audit"]["structural_eligibility"])
        self.assertFalse(result["structural_test_audit"]["macro_expansion_resolved"])
        self.assertEqual(result["physical_whole_file_diagnostic"]["crate"]["lines"], 3)

    def test_unmapped_module_parent_is_audited_without_adding_a_coverage_line(self):
        """An unreported parent can still make a measured child's cfg/path ambiguous."""

        self.native["data"][0]["files"] = [item for item in self.native["data"][0]["files"]
                                           if item["filename"] != str(self.crate / "src/lib.rs")]
        before = self.result()
        self.assertTrue(before["structural_test_audit"]["structural_eligibility"])
        self.assertIn("src/lib.rs", before["structural_test_audit"]["inspected_production_scope_files"])
        self.classification["diagnostics"].append({"file": "src/lib.rs", "code": "ambiguous_module_path"})
        after = self.result()
        self.assertFalse(after["structural_test_audit"]["structural_eligibility"])
        self.assertEqual(before["physical_whole_file_diagnostic"], after["physical_whole_file_diagnostic"])

    def test_unreported_parent_contexts_and_regions_fail_closed_in_the_direct_helper(self):
        """No-row parents receive the same context/region/line validation as rows."""

        baseline = copy.deepcopy(self.classification)
        production = set(self.targets[0]["inputs"])
        for case in ("missing_contexts", "not_contexts", "empty_contexts", "context_type", "missing_regions",
                     "empty_regions", "region_type", "root_kind", "root_span", "test_bool", "line_gap"):
            with self.subTest(case=case):
                self.classification = copy.deepcopy(baseline)
                entry = self.production()
                if case == "missing_contexts": del entry["contexts"]
                elif case == "not_contexts": entry["contexts"] = {}
                elif case == "empty_contexts": entry["contexts"] = []
                elif case == "context_type": entry["contexts"] = [None]
                elif case == "missing_regions": del entry["contexts"][0]["regions"]
                elif case == "empty_regions": entry["contexts"][0]["regions"] = []
                elif case == "region_type": entry["contexts"][0]["regions"] = [None]
                elif case == "root_kind": entry["contexts"][0]["regions"][0]["kind"] = "item"
                elif case == "root_span": entry["contexts"][0]["regions"][0]["span"]["start"]["byte"] = 1
                elif case == "test_bool": entry["contexts"][0]["regions"][0]["test_only"] = 1
                else: entry["lines"] = []
                with self.assertRaises(ValueError):
                    E._structural_test_audit([], self.classification, {}, production, self.snapshot)

    def test_malformed_audit_sites_and_false_external_declarations_fail_closed(self):
        """A malformed record cannot promote a body by spelling it a declaration."""

        self.site()
        baseline = copy.deepcopy(self.classification)
        for case in ("version_bool", "complete_bool", "expanded", "missing_sites", "unknown_owner", "unknown_disposition",
                     "outside_owner", "empty_attribute", "false_external", "unbound_external"):
            with self.subTest(case=case):
                self.classification = copy.deepcopy(baseline)
                audit = self.production()["test_audit"]
                site = audit["sites"][0]
                if case == "version_bool": audit["schema_version"] = True
                elif case == "complete_bool": audit["complete"] = 1
                elif case == "expanded": audit["macro_expansion_resolved"] = True
                elif case == "missing_sites": del audit["sites"]
                elif case == "unknown_owner": site["owner_kind"] = "unknown"
                elif case == "unknown_disposition": site["disposition"] = "ignore"
                elif case == "outside_owner": site["span"]["start"]["byte"] = 1
                elif case == "empty_attribute": site["attribute_span"]["end"] = copy.deepcopy(site["attribute_span"]["start"])
                elif case == "false_external": site["disposition"] = "external_declaration"
                else: site.update(owner_kind="external_mod", disposition="external_declaration")
                with self.assertRaises(ValueError): self.result()


@unittest.skipUnless(os.environ.get("TFM_WHOLE_FILE_COVERAGE_NATIVE_TESTS") == "1", "opt-in tiny pinned compiler/classifier fixture")
class NativeWholeFileTests(unittest.TestCase):
    """Native inputs veto candidates; exact test-only macro ownership can qualify."""

    def test_native_structural_diagnostic_cannot_promote_macro_generated_tests(self):
        """Parsed eligibility distinguishes visible bodies but cannot prove macro origins."""

        with tempfile.TemporaryDirectory(prefix="tfm-structural-native-") as directory:
            root = Path(directory).resolve()
            environment = {key: value for key, value in os.environ.items() if key not in (
                "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
                "CARGO_TARGET_DIR", "CARGO_BUILD_TARGET", "LLVM_COV", "LLVM_PROFDATA", "LLVM_PROFILE_FILE")}
            environment.update(CARGO_INCREMENTAL="0", CARGO_TERM_COLOR="never")
            B.producer_from_version(subprocess.check_output(["rustc", "+nightly", "-Vv"], env=environment, text=True, timeout=30))
            manifest = SCRIPT.parent / "coverage-layout/Cargo.toml"
            classifier_sources = {str(path): fingerprint_file(path) for path in [manifest, manifest.with_name("Cargo.lock"),
                *sorted(manifest.parent.joinpath("src").glob("*.rs"))]}
            build = subprocess.run(["cargo", "+nightly", "build", "--manifest-path", str(manifest), "--locked", "--offline",
                                    "--target-dir", str(root / "classifier")], env=environment, capture_output=True, text=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            self.assertNotIn("warning:", build.stderr)
            for path, identity in classifier_sources.items(): self.assertEqual(fingerprint_file(Path(path)), identity)
            executable = root / "classifier/debug/transformers-coverage-layout"
            unit = '#[test] fn check() { assert_eq!(super::value(), 1); }\n'
            runtime = 'pub fn value() -> u32 { 1 }\n'
            declaration = '#[cfg(test)] mod unit;\n'
            variants = {
                "inline": (runtime + '#[cfg(test)] mod unit { ' + unit + '}\n', unit, {}, False),
                "extracted": (runtime + declaration, unit, {}, True),
                "shared_line": (runtime.rstrip() + ' ' + declaration, unit, {}, True),
                "runtime_macro_cfg": ('macro_rules! scalar { () => { 1 }; }\n#[cfg(feature="fixture")] pub fn value() -> u32 { scalar!() }\n' + declaration, unit, {}, True),
                "generated_test": (runtime + declaration + 'macro_rules! hidden { () => { #[test] fn generated_test() { assert_eq!(value(), 1); } }; } hidden!();\n', unit, {}, True),
                "generated_helper": (runtime + declaration + 'macro_rules! hidden { () => { #[cfg(test)] fn generated_helper() -> u32 { 1 } }; } hidden!();\n',
                                     unit + '#[test] fn helper() { assert_eq!(super::generated_helper(), 1); }\n', {}, True),
                "bare_test": (runtime + declaration + '#[test] fn residual() { assert_eq!(value(), 1); }\n', unit, {}, False),
                "inherent": (runtime + declaration + 'struct S; impl S { #[cfg(test)] fn helper() -> u32 { 1 } }\n',
                             unit + '#[test] fn helper() { assert_eq!(super::S::helper(), 1); }\n', {}, False),
                "expression_statement": ('pub fn value() -> u32 { #[cfg(test)] let _x = 1; #[cfg(test)] { assert_eq!(1, 1); } 1 }\n' + declaration, unit, {}, False),
                "conditional_attribute": ('#[cfg_attr(test, inline)] ' + runtime + declaration, unit, {}, False),
                "uncertain_cfg": ('#[cfg(any(test, feature="fixture"))] ' + runtime + declaration, unit, {}, False),
                "conditional_path": (runtime + '#[cfg(test)] #[cfg_attr(test, path="unit.rs")] mod unit;\n', unit, {}, False),
                "runtime_include": ('include!("runtime.rs");\n' + declaration, unit, {"src/runtime.rs": runtime}, False),
                "test_include": (runtime + declaration, unit + 'include!("child.rs");\n#[test] fn child() { assert_eq!(child_value(), 1); }\n',
                                 {"src/child.rs": 'fn child_value() -> u32 { 1 }\n'}, False),
                "raw_production_input": (runtime + declaration + '#[cfg(test)] mod support; const RAW: &str = include_str!("support.rs");\n',
                                         unit + '#[test] fn helper() { assert_eq!(super::support::helper(), 1); }\n',
                                         {"src/support.rs": 'pub fn helper() -> u32 { 1 }\n'}, False),
            }
            for name, (source, tests, extra, expected) in variants.items():
                with self.subTest(name=name):
                    crate = root / name
                    put(crate / "Cargo.toml", '[package]\nname="structural_fixture"\nversion="0.1.0"\nedition="2024"\n[workspace]\n[features]\nfixture=[]\n')
                    put(crate / "Cargo.lock", 'version=4\n[[package]]\nname="structural_fixture"\nversion="0.1.0"\n')
                    put(crate / "src/lib.rs", '#![allow(dead_code)]\n' + source)
                    put(crate / "src/unit.rs", tests)
                    for path, content in extra.items(): put(crate / path, content)
                    put(crate / "tests/integration.rs", '#[test] fn check() { assert_eq!(structural_fixture::value(), 1); }\n')
                    with mock.patch.object(check_release, "CRATE", crate), mock.patch.object(check_release, "FEATURES", "fixture"), \
                            mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(sys, "argv", [
                                "check_release", "--gate", "coverage", "--toolchain", "nightly", "--offline", "--coverage-compiler-evidence",
                            ]), mock.patch("builtins.print"):
                        code = check_release.main()
                    results_path = next((crate / "target/release-gates").glob("*/results.json"))
                    output, results = results_path.parent, E.read_json(results_path)
                    self.assertEqual(code, 0, json.dumps(results))
                    log = output / results["gates"][0]["log"]
                    self.assertIn("test unit::check ... ok", log.read_text())
                    setup = E.read_json(output / "compiler-capture-setup.json")
                    before = E.read_json(output / "coverage-inputs-before.json")["sources"]
                    binding = B.validate_capture(setup, before, crate)
                    E.capture_classifier(crate, before, executable, output / "classifier", E.target_entries(binding["targets"], crate))
                    result = E.qualify_capture(setup, before, output / "classifier/invocation.json", crate,
                                               coverage_log=log, run_results=results_path)
                    self.assertEqual(result["structural_test_audit"]["structural_eligibility"], expected, json.dumps(result["structural_test_audit"]))
                    self.assertFalse(result["structural_test_audit"]["production_measurement_qualified"])
                    self.assertFalse(result["production_only_measurement"])
                    self.assertFalse(result["production_only_exclusions_qualified"])
                    self.assertNotIn("physical_scoped_production", result)
                    self.assertFalse(result["compiler_dependency_closure_verified"])
                    self.assertFalse(result["release_gate_promotion"])
                    self.assertTrue(all(item["meets_threshold"] for item in result["physical_whole_file_diagnostic"].values()))
                    if not expected: self.assertTrue(result["structural_test_audit"]["blockers"])
                    if name == "raw_production_input":
                        self.assertIn("retained_only_test_module_contexts", [item["code"] for item in result["structural_test_audit"]["blockers"]])
                    if name == "generated_test": self.assertIn("test generated_test ... ok", log.read_text())
                    if name == "generated_helper": self.assertIn("test unit::helper ... ok", log.read_text())

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
            put(crate / "Cargo.toml", '[package]\nname="whole_file_fixture"\nversion="0.1.0"\nedition="2024"\n[workspace]\n[features]\nfixture=[]\nunselected=[]\n[dev-dependencies]\nproptest={version="=1.11.0",default-features=false,features=["std"]}\n')
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
#[cfg(test)] mod whole_macro;
#[cfg(test)] #[path = "whole_macro.rs"] mod whole_macro_alias;
#[cfg(test)] mod property_tests;
#[cfg(test)] mod independent_child;
pub fn value() -> u32 { production::value() + production_alias::value() + macro_live::value() + included::value() }
''')
            for name in ("tests/production.rs", "aliased.rs", "macro_module.rs", "included.rs", "raw_input.rs", "uncertain.rs", "conditional.rs"):
                put(crate / "src" / name, 'pub fn value() -> u32 { 1 }\n#[test] fn own_test() { assert_eq!(value(), 1); }\n')
            with (crate / "src/macro_module.rs").open("a") as stream:
                stream.write('macro_rules! live { () => { pub fn live_value() -> u32 { 2 } }; }\nlive!();\n#[test] fn live_test() { assert_eq!(live_value(), 2); }\n')
            for name in ("unit only.rs", "feature_test.rs"):
                put(crate / "src" / name, '#[test] fn check() { assert_eq!(super::value(), 4); }\n')
            put(crate / "src/whole_macro.rs", '''macro_rules! generated_test {
    () => { #[test] fn generated() { assert_eq!(super::value(), 4); } };
}
generated_test!();
include!("opaque_child.rs");
mod independently_owned { include!("independent_child.rs"); }
#[test] fn children() {
    assert_eq!(child_value(), 3);
    assert_eq!(independently_owned::independent_value(), 5);
}
''')
            put(crate / "src/opaque_child.rs", 'pub fn child_value() -> u32 { 3 }\n')
            put(crate / "src/independent_child.rs", 'pub fn independent_value() -> u32 { 5 }\n#[test] fn own_test() { assert_eq!(independent_value(), 5); }\n')
            put(crate / "src/property_tests.rs", '''fn round_trip(value: u8) -> u8 { u16::from(value) as u8 }
proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(16))]
    #[test]
    fn property_round_trip(value in 0_u8..=255) {
        proptest::prop_assert_eq!(round_trip(value), value);
    }
}
''')
            # An exact macro span cannot heal unknown cfg or conditional paths.
            for name in ("uncertain.rs", "conditional.rs"):
                with (crate / "src" / name).open("a") as stream:
                    stream.write('macro_rules! generated { () => { #[test] fn macro_check() { assert_eq!(value(), 1); } }; }\ngenerated!();\n')
            put(crate / "tests/integration.rs", '#[test] fn check() { assert_eq!(whole_file_fixture::value(), 4); }\n')
            lock = subprocess.run(["cargo", "+nightly", "generate-lockfile", "--manifest-path", str(crate / "Cargo.toml"),
                                   "--offline"], env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(lock.returncode, 0, lock.stdout + lock.stderr)
            # Use the real runner, not a synthesized success manifest. Its exact
            # indexed command log and retained native status are part of proof.
            with mock.patch.object(check_release, "CRATE", crate), mock.patch.object(check_release, "FEATURES", "fixture"), \
                    mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(sys, "argv", [
                        "check_release", "--gate", "coverage", "--toolchain", "nightly", "--offline", "--coverage-compiler-evidence",
                    ]), mock.patch("builtins.print"):
                code = check_release.main()
            results_path = next((crate / "target/release-gates").glob("*/results.json"))
            output = results_path.parent
            results = E.read_json(results_path)
            self.assertEqual(code, 0, json.dumps(results))
            log = output / results["gates"][0]["log"]
            self.assertIn("test property_tests::property_round_trip ... ok", log.read_text())
            setup = E.read_json(output / "compiler-capture-setup.json")
            before = E.read_json(output / "coverage-inputs-before.json")["sources"]
            require_same_sources(before, capture_sources(crate))
            binding = B.validate_capture(setup, before, crate)
            capture_dir = output / "classifier"
            E.capture_classifier(crate, before, executable, capture_dir, E.target_entries(binding["targets"], crate))
            result = E.qualify_capture(setup, before, capture_dir / "invocation.json", crate,
                                       coverage_log=log, run_results=results_path)
            rows = {row["path"]: row for row in result["files"]}
            classification = E.read_json(capture_dir / "classification.json")
            self.assertEqual({name for name, row in rows.items() if row["excluded"]}, {
                "src/unit only.rs", "src/feature_test.rs", "src/whole_macro.rs", "src/property_tests.rs", "src/independent_child.rs"},
                json.dumps({"rows": {name: row["reason"] for name, row in rows.items()},
                            "diagnostics": classification["diagnostics"]}))
            self.assertEqual(rows["src/opaque_child.rs"]["reason"], "retained_unclassified_source")
            self.assertGreater(rows["src/opaque_child.rs"]["mapped_lines"], 0)
            for name in ("tests/production.rs", "aliased.rs", "macro_module.rs", "included.rs", "raw_input.rs"):
                self.assertEqual(rows["src/" + name]["reason"], "retained_production_compiler_input")
            for name in ("uncertain.rs", "conditional.rs"):
                self.assertEqual(rows["src/" + name]["reason"], "retained_unknown_mixed_or_non_test_file_ownership")
            for name in ("whole_macro.rs", "property_tests.rs"):
                file = next(item for item in classification["files"] if item["path"] == "src/" + name)
                issues = [issue for issue in classification["diagnostics"]
                          if issue["file"] == file["path"] and issue["code"] == "opaque_macro_or_verbatim"]
                self.assertTrue(issues)
                for context in file["contexts"]:
                    self.assertTrue(context["regions"][0]["test_only"])
                    for issue in issues:
                        self.assertTrue(any(region["kind"] == "macro" and region["test_only"] and region["span"] == issue["span"]
                                            for region in context["regions"]))
                if name == "whole_macro.rs": self.assertGreaterEqual(len(file["contexts"]), 2)
            self.assertTrue(result["whole_test_file_exclusions_qualified"])
            self.assertFalse(result["production_only_exclusions_qualified"])
            self.assertFalse(result["release_gate_promotion"])
            clean_sibling = output / "clean-sibling.log"
            put(clean_sibling, log.read_text())
            with log.open("a") as stream: stream.write("warning: mismatched profile fixture\n")
            with self.assertRaisesRegex(ValueError, "actual retained command log"):
                E.qualify_capture(setup, before, capture_dir / "invocation.json", crate,
                                  coverage_log=clean_sibling, run_results=results_path)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                E.qualify_capture(setup, before, capture_dir / "invocation.json", crate,
                                  coverage_log=log, run_results=results_path)
            # Even a modified manifest claiming the warning-bearing bytes are
            # original cannot bypass the independent strict diagnostic reader.
            results["gates"][0]["log_identity"] = fingerprint_file(log)
            put(results_path, json.dumps(results))
            with self.assertRaisesRegex(ValueError, "unqualified coverage diagnostic"):
                E.qualify_capture(setup, before, capture_dir / "invocation.json", crate,
                                  coverage_log=log, run_results=results_path)


if __name__ == "__main__":
    unittest.main()
