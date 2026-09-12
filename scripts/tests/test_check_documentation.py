"""Source inventory and enforcement boundaries; no compiler or network required."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "check_documentation.py"
SPEC = importlib.util.spec_from_file_location("check_documentation", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class DocumentationWrapperTests(unittest.TestCase):
    """Keep authored discovery conservative and bind every parsed source snapshot."""

    def setUp(self):
        """Allocate isolated source and helper artifacts without compiling them."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.write("src/lib.rs", "//! Root.\n/// Function.\nfn documented() {}\n")
        self.classifier = self.write("parser", "test executable fingerprint")

    def write(self, name, contents):
        """Write only paths owned by this test's temporary source tree."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        return path

    def report(self, paths, *, passed=True, complete=True):
        """Supply the parser's external snapshot contract, independently of discovery."""
        return {
            "schema_version": 1, "root": str(self.root), "passed": passed,
            "complete": complete, "files": [
                {"path": path, "sha256": AUDIT.digest(self.root / path),
                 "sites": [], "macros": [], "inline_test_bodies": []}
                for path in paths
            ],
        }

    def test_discovery_does_not_trust_test_filenames_or_audit_retained_evidence(self):
        """Only explicit discovery roots are skipped; a *_tests name proves nothing."""
        names = ["src/lib.rs", "src/orphan_tests.rs", "examples/demo.rs",
                 "scripts/tool/src/main.rs", "scripts/tests/fixtures/old.rs",
                 "tests/harness.rs", "acceptance/history/old.rs", "models/copied.rs"]
        completed = subprocess.CompletedProcess([], 0, stdout="\0".join(names).encode())
        with mock.patch.object(AUDIT.subprocess, "run", return_value=completed):
            sources, excluded = AUDIT.discover_sources(self.root)
        self.assertEqual(sources, sorted(names[:4]))
        self.assertEqual({item["path"] for item in excluded}, set(names[4:]))

    def test_unreferenced_source_is_added_as_a_conservative_production_entry(self):
        """A newly authored source cannot disappear merely because its module is unwired."""
        self.write("src/orphan_tests.rs", "fn undocumented_helper() {}\n")
        sources = ["src/lib.rs", "src/orphan_tests.rs"]
        with mock.patch.object(AUDIT, "discover_sources", return_value=(sources, [])), \
             mock.patch.object(AUDIT, "invoke", side_effect=[self.report(sources[:1]), self.report(sources)]) as invoke:
            result = AUDIT.audit(self.root, self.classifier)
        self.assertEqual(invoke.call_args_list[1].args[2], sources)
        self.assertEqual(result["inventory"]["conservative_orphan_entries"], [sources[1]])

    def test_production_reference_into_fixture_directory_is_still_audited(self):
        """Discovery exclusions cannot override the parser's actual module graph."""
        self.write("scripts/tests/fixtures/reused.rs", "fn helper() {}\n")
        report = self.report(["src/lib.rs", "scripts/tests/fixtures/reused.rs"])
        report["files"][1]["sites"] = [{
            "name": "helper", "kind": "free_function", "test_only": False,
            "style_present": False, "span": {"start": {"line": 1}},
            "required_style": "///", "rustdoc_present": False,
            "computed_documentation": False, "origin": "parsed_item",
        }]
        with mock.patch.object(AUDIT, "discover_sources", return_value=(["src/lib.rs"], [])), \
             mock.patch.object(AUDIT, "invoke", return_value=report):
            result = AUDIT.audit(self.root, self.classifier)
        self.assertEqual(result["inventory"]["missing"][0]["path"], "scripts/tests/fixtures/reused.rs")
        self.assertEqual(result["inventory"]["missing"][0]["component"], "tooling")

    def test_stale_parser_snapshot_and_outside_paths_fail_closed(self):
        """A plausible report cannot bind mismatched bytes or an escaping path."""
        with mock.patch.object(AUDIT, "discover_sources", return_value=(["src/lib.rs"], [])):
            for path, checksum in [("src/lib.rs", "0" * 64), ("../outside.rs", "0" * 64)]:
                report = self.report(["src/lib.rs"])
                report["files"][0].update(path=path, sha256=checksum)
                with self.subTest(path=path), mock.patch.object(AUDIT, "invoke", return_value=report):
                    with self.assertRaisesRegex(ValueError, "snapshot"):
                        AUDIT.audit(self.root, self.classifier)

    def test_source_edit_during_parsing_invalidates_the_inventory(self):
        """Read-only checks fail if another process edits a source during inspection."""
        report = self.report(["src/lib.rs"])

        def changed(*_, **__):
            self.write("src/lib.rs", "fn changed() {}\n")
            return report

        with mock.patch.object(AUDIT, "discover_sources", return_value=(["src/lib.rs"], [])), \
             mock.patch.object(AUDIT, "invoke", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed during"):
                AUDIT.audit(self.root, self.classifier)

    def test_helper_schema_and_root_are_verified(self):
        """An incompatible helper or report from another checkout is rejected."""
        for schema, root in [(2, str(self.root)), (1, str(self.root / "other"))]:
            completed = subprocess.CompletedProcess([], 0, stdout=json.dumps({
                "schema_version": schema, "root": root}).encode())
            with self.subTest(schema=schema, root=root), \
                 mock.patch.object(AUDIT.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(ValueError, "schema or another source root"):
                    AUDIT.invoke(self.classifier, self.root, ["src/lib.rs"])

    def test_helper_failure_preserves_its_actionable_diagnostic(self):
        """Expose the parser's source-change error instead of only its invocation."""
        failure = subprocess.CalledProcessError(2, ["parser"], stderr=b"src/lib.rs changed between parses")
        with mock.patch.object(AUDIT.subprocess, "run", side_effect=failure):
            with self.assertRaisesRegex(ValueError, "src/lib.rs changed between parses"):
                AUDIT.invoke(self.classifier, self.root, ["src/lib.rs"])

    def test_cli_distinguishes_style_failure_from_incomplete_inspection(self):
        """Inventory mode relaxes style enforcement, never unresolved source inspection."""
        arguments = ["--root", str(self.root), "--classifier", str(self.classifier)]
        for complete, passed, inventory, expected in [
            (True, True, False, 0), (True, False, False, 1),
            (True, False, True, 0), (False, False, True, 2),
        ]:
            report = self.report([], passed=passed, complete=complete)
            report["inventory"] = {"groups": {}}
            with self.subTest(complete=complete, passed=passed, inventory=inventory), \
                 mock.patch.object(AUDIT, "audit", return_value=report), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                actual = AUDIT.main(arguments + (["--inventory-only"] if inventory else []))
            self.assertEqual(actual, expected)

    def test_standards_discovers_integration_and_tool_tests_but_preserves_evidence(self):
        """Strict inventory includes authored harnesses, with fixtures still explicit exclusions."""
        names = ["src/lib.rs", "tests/harness.rs", "tests/support/data.rs",
                 "scripts/tool/tests/integration.rs", "scripts/tests/fixtures/old.rs",
                 "acceptance/history/old.rs"]
        completed = subprocess.CompletedProcess([], 0, stdout="\0".join(names).encode())
        for name in names:
            self.write(name, "//! Source.\n")
        with mock.patch.object(AUDIT.subprocess, "run", return_value=completed):
            sources, excluded = AUDIT.discover_sources(self.root, standards=True)
        self.assertEqual(sources, sorted(names[:4]))
        self.assertEqual({item["path"] for item in excluded}, set(names[4:]))
        self.assertEqual(AUDIT.default_entries(self.root, sources, standards=True),
                         sorted([names[0], names[1], names[3]]))

    def test_standards_invocation_requires_schema_two_without_changing_default(self):
        """Strict dispatch requests its own version and rejects legacy helper output."""
        for standards, schema, expected_flag in [
            (False, 1, "--documentation"), (True, 2, "--standards")
        ]:
            completed = subprocess.CompletedProcess([], 0, stdout=json.dumps({
                "schema_version": schema, "root": str(self.root)}).encode())
            with mock.patch.object(AUDIT.subprocess, "run", return_value=completed) as run:
                AUDIT.invoke(self.classifier, self.root, ["src/lib.rs"], standards=standards)
            self.assertEqual(run.call_args.args[0][1], expected_flag)
        legacy = subprocess.CompletedProcess([], 0, stdout=json.dumps({
            "schema_version": 1, "root": str(self.root)}).encode())
        with mock.patch.object(AUDIT.subprocess, "run", return_value=legacy):
            with self.assertRaisesRegex(ValueError, "unsupported schema"):
                AUDIT.invoke(self.classifier, self.root, ["src/lib.rs"], standards=True)

    def test_standards_counts_members_and_test_sites_without_function_misclassification(self):
        """Strict findings identify declaration kinds and include opaque test macro items."""
        self.write("tests/harness.rs", "//! Tests.\n")
        report = self.report(["src/lib.rs", "tests/harness.rs"], passed=False, complete=False)
        report["schema_version"] = 2
        report["files"][1]["sites"] = [{
            "name": "payload", "kind": "field", "test_only": True,
            "style_present": False, "span": {"start": {"line": 2}},
            "required_style": "///", "rustdoc_present": False,
            "computed_documentation": False, "origin": "parsed_item",
        }]
        report["files"][1]["macros"] = [{"name": "template", "test_only": True,
            "unresolved_functions": [{"start": {"line": 3}}],
            "unresolved_items": [{"start": {"line": 4}}]}]
        with mock.patch.object(AUDIT, "discover_sources", return_value=(["src/lib.rs", "tests/harness.rs"], [])), \
             mock.patch.object(AUDIT, "invoke", return_value=report):
            result = AUDIT.audit(self.root, self.classifier, standards=True)
        counts = result["inventory"]["groups"]["integration_tests"]
        self.assertEqual(counts["authored_sites"], 1)
        self.assertEqual(counts["test_sites"], 1)
        self.assertEqual(counts["missing_style"], 1)
        self.assertEqual(counts["by_kind"], {"field": {"sites": 1, "missing": 1}})
        self.assertNotIn("production_functions", counts)
        self.assertTrue(result["inventory"]["missing"][0]["test_only"])
        self.assertEqual(result["inventory"]["unresolved_macro_items"][0]["line"], 4)
        self.assertEqual(counts["unresolved_macro_functions"], 1)

    def test_standards_cli_keeps_style_and_completeness_exit_codes(self):
        """Opt-in summaries and dispatch preserve the established three-way exit contract."""
        arguments = ["--root", str(self.root), "--classifier", str(self.classifier), "--standards"]
        report = self.report([], passed=False)
        report["schema_version"] = 2
        report["inventory"] = {"groups": {"shared": {"authored_sites": 2, "test_sites": 1,
            "missing_style": 1, "inline_test_bodies": 0,
            "unresolved_macro_functions": 0, "unresolved_macro_items": 0}}}
        with mock.patch.object(AUDIT, "audit", return_value=report) as audit, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(AUDIT.main(arguments), 1)
        self.assertTrue(audit.call_args.kwargs["standards"])
        self.assertIn("1/2 authored sites", output.getvalue())


if __name__ == "__main__":
    unittest.main()
