"""Bounded byte identity tests; no compiler, model artifacts or network required."""

import copy
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "coverage_provenance.py"
SPEC = importlib.util.spec_from_file_location("coverage_provenance", SCRIPT)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class SourceSnapshotTests(unittest.TestCase):
    """Exercise exact bytes, changed inventories, resource bounds and unsafe inputs."""

    def setUp(self):
        """Create a minimal independent crate-shaped fixture."""

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "src").mkdir()
        (self.root / "Cargo.toml").write_text("[package]\nname='fixture'\n")
        (self.root / "Cargo.lock").write_text("version = 4\n")
        (self.root / "src/lib.rs").write_text("pub fn value() -> u32 { 7 }\n")

    def test_exact_hash_line_conventions_and_empty_files(self):
        """Hash original bytes, including BOM/CRLF and a non-LF final line."""

        for data, lines in ((b"", 0), (b"a", 1), (b"a\n", 1), (b"a\r\n\r\nlast", 3), ("\ufeffé🙂\n".encode(), 1)):
            with self.subTest(data=data):
                source = self.root / "src/lib.rs"
                source.write_bytes(data)
                value = P.fingerprint_file(source)
                self.assertEqual(value, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "lines": lines})

    def test_stream_boundaries_preserve_hashes_and_lf_counts(self):
        """Chunking cannot normalize or lose bytes at a buffer boundary."""

        data = b"x" * (P.HASH_BUFFER_BYTES - 1) + b"\r\n" + b"y" * P.HASH_BUFFER_BYTES
        path = self.root / "src/lib.rs"
        path.write_bytes(data)
        identity = P.fingerprint_file(path, maximum_bytes=len(data))
        self.assertEqual(identity["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(identity["lines"], 2)
        with self.assertRaisesRegex(P.CoverageProvenanceError, "byte bound"):
            P.fingerprint_file(path, maximum_bytes=len(data) - 1)

    def test_ignored_source_is_included_but_external_models_are_not_read(self):
        """A Git-ignored module may compile; an artifact root is not source input."""

        (self.root / ".gitignore").write_text("src/ignored.rs\n")
        (self.root / "src/ignored.rs").write_text("fn ignored_by_git() {}")
        (self.root / "models").symlink_to(self.root / "not-mounted", target_is_directory=True)
        (self.root / "target").mkdir()
        (self.root / "target/out.rs").write_text("generated artifact outside policy")
        snapshot = P.capture_sources(self.root)
        self.assertIn("src/ignored.rs", snapshot["files"])
        self.assertNotIn("target/out.rs", snapshot["files"])
        self.assertFalse(snapshot["compiler_dependency_closure_verified"])
        self.assertEqual(snapshot["total_bytes"], sum(entry["bytes"] for entry in snapshot["files"].values()))

    def test_inventory_digest_is_portable_sorted_and_byte_bound(self):
        """Equivalent inputs in another checkout have the same content digest."""

        first = P.capture_sources(self.root)
        with tempfile.TemporaryDirectory() as other:
            other = Path(other).resolve()
            for name in reversed(list(first["files"])):
                target = other / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((self.root / name).read_bytes())
            second = P.capture_sources(other)
            self.assertEqual(first["inventory_sha256"], second["inventory_sha256"])
            with self.assertRaisesRegex(P.CoverageProvenanceError, "root changed"):
                P.require_same_sources(first, second)
        P.require_same_sources(first, P.capture_sources(self.root))

    def test_same_length_edit_added_removed_and_renamed_files_are_rejected(self):
        """Content hashes and membership, not size or timestamps alone, bind inputs."""

        baseline = P.capture_sources(self.root)
        path = self.root / "src/lib.rs"
        original = path.read_bytes()
        path.write_bytes(original.replace(b"7", b"8"))
        with self.assertRaisesRegex(P.CoverageProvenanceError, "changed=.*src/lib.rs"):
            P.require_same_sources(baseline, P.capture_sources(self.root))
        path.write_bytes(original)
        extra = self.root / "src/extra.rs"
        extra.write_text("fn extra() {}")
        with self.assertRaisesRegex(P.CoverageProvenanceError, "added=.*extra.rs"):
            P.require_same_sources(baseline, P.capture_sources(self.root))
        before_rename = P.capture_sources(self.root)
        extra.rename(self.root / "src/renamed.rs")
        with self.assertRaisesRegex(P.CoverageProvenanceError, "removed=.*extra.rs"):
            P.require_same_sources(before_rename, P.capture_sources(self.root))

    def test_touch_without_byte_changes_preserves_content_identity(self):
        """Post-run content comparison does not fail merely because mtimes differ."""

        before = P.capture_sources(self.root)
        path = self.root / "src/lib.rs"
        value = path.stat()
        os.utime(path, ns=(value.st_atime_ns, value.st_mtime_ns + 1_000_000))
        P.require_same_sources(before, P.capture_sources(self.root))
        after = copy.deepcopy(before)
        after["inventory_sha256"] = "0" * 64
        with self.assertRaisesRegex(P.CoverageProvenanceError, "digest changed"):
            P.require_same_sources(before, after)

    def test_each_resource_bound_and_invalid_bound_fail_closed(self):
        """No unbounded input walk or file read can be mistaken for a snapshot."""

        for options, message in (({"maximum_files": 2}, "file bound"), ({"maximum_total_bytes": 1}, "total byte"), ({"maximum_file_bytes": 1}, "byte bound")):
            with self.subTest(options=options), self.assertRaisesRegex(P.CoverageProvenanceError, message):
                P.capture_sources(self.root, **options)
        for bound in (0, -1, True, 1.5):
            with self.subTest(bound=bound), self.assertRaises(P.CoverageProvenanceError):
                P.capture_sources(self.root, maximum_files=bound)
            with self.assertRaises(P.CoverageProvenanceError):
                P.fingerprint_file(self.root / "src/lib.rs", maximum_bytes=bound)
        with mock.patch.object(P, "MAX_DIRECTORY_DEPTH", 0):
            (self.root / "src/nested").mkdir()
            with self.assertRaisesRegex(P.CoverageProvenanceError, "traversal bound"):
                P.capture_sources(self.root)

    def test_root_inputs_and_test_fixture_bytes_are_fingerprinted(self):
        """Build configuration and acceptance fixtures can alter a test execution."""

        for name in ("tests/config.json", "examples/demo.rs", "benches/sample.rs", "acceptance/record.json", "build.rs", "rust-toolchain.toml", ".cargo/config.toml"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture\n")
        snapshot = P.capture_sources(self.root)
        for name in ("tests/config.json", "examples/demo.rs", "benches/sample.rs", "acceptance/record.json", "build.rs", "rust-toolchain.toml", ".cargo/config.toml"):
            self.assertEqual(snapshot["files"][name]["bytes"], 8)
        (self.root / "Cargo.lock").unlink()
        with self.assertRaisesRegex(P.CoverageProvenanceError, "mandatory"):
            P.capture_sources(self.root)

    def test_script_inputs_are_bound_but_generated_outputs_are_scoped_out(self):
        """Rust include_bytes fixtures include scripts, but Python creates bytecode."""

        scripts = self.root / "scripts"
        scripts.mkdir()
        (scripts / "reference.py").write_text("print('fixture')\n")
        before = P.capture_sources(self.root)
        self.assertIn("scripts/reference.py", before["files"])
        for name in P.SCRIPT_OUTPUT_DIRECTORIES:
            directory = scripts / name
            directory.mkdir()
            (directory / "generated").write_bytes(b"generated")
        P.require_same_sources(before, P.capture_sources(self.root))
        (scripts / "reference.py").write_text("print('changed')\n")
        with self.assertRaisesRegex(P.CoverageProvenanceError, "reference.py"):
            P.require_same_sources(before, P.capture_sources(self.root))
        # The same spelling in Rust input roots is not an exclusion.
        (self.root / "src/target").mkdir()
        (self.root / "src/target/code.rs").write_text("fn input() {}")
        self.assertIn("src/target/code.rs", P.capture_sources(self.root)["files"])

    def test_directory_listing_is_bounded_before_sorting_or_hashing(self):
        """Reject oversized inventories without materializing the full iterator."""

        real_scandir = os.scandir

        @contextmanager
        def entries(path):
            """Produce more entries than the budget; reject before the sentinel."""

            if path == self.root / "src":
                def bounded():
                    """Raise if the walker requests names beyond the budget."""

                    for _ in range(7):
                        yield SimpleNamespace(name="lib.rs")
                    self.fail("unbounded directory iteration")

                yield bounded()
            else:
                with real_scandir(path) as directory:
                    yield directory

        with mock.patch.object(P.os, "scandir", entries), self.assertRaisesRegex(P.CoverageProvenanceError, "directory entry bound"):
            P.capture_sources(self.root, maximum_files=3)

    def test_symlinked_and_nonregular_inputs_are_rejected(self):
        """Reject aliases rather than reading outside or omitting dangling inputs."""

        for relative, target in (("src/link.rs", "lib.rs"), ("src/link.rs", "missing.rs"), ("tests", "src"), (".cargo", "missing-directory")):
            path = self.root / relative
            path.symlink_to(target)
            try:
                with self.assertRaises(P.CoverageProvenanceError):
                    P.capture_sources(self.root)
            finally:
                path.unlink()
        path = self.root / "src/pipe.rs"
        os.mkfifo(path)
        with self.assertRaises(P.CoverageProvenanceError):
            P.capture_sources(self.root)
        with self.assertRaises(P.CoverageProvenanceError):
            P.fingerprint_file(path)

    def test_mutation_during_read_is_detected_on_the_open_descriptor(self):
        """A same-size edit during hashing must not produce accepted mixed bytes."""

        path = self.root / "src/lib.rs"
        before = path.stat()
        changed = SimpleNamespace(**{name: getattr(before, name) for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")})
        changed.st_mtime_ns += 1
        with mock.patch.object(P.os, "fstat", side_effect=[before, changed]), self.assertRaisesRegex(P.CoverageProvenanceError, "during read"):
            P.fingerprint_file(path)
        with mock.patch.object(P.os, "fstat", return_value=changed), self.assertRaisesRegex(P.CoverageProvenanceError, "before read"):
            P.fingerprint_file(path)

    def test_relative_path_policy_rejects_ambiguous_spellings(self):
        """Do not give traversal, Windows prefixes or separators alternate names."""

        for path in (None, "", "/src/a.rs", "src/../a.rs", "src/./a.rs", "src//a.rs", "C:/a.rs", "src\\a.rs", "src/a\0.rs", "models/a.rs"):
            with self.subTest(path=path):
                self.assertFalse(P._in_scope(path))
        self.assertTrue(P._in_scope("src/é.rs"))

    def classifier_report(self, snapshot):
        """Construct only classifier identity fields; no exclusion proof is implied."""

        entry = snapshot["files"]["src/lib.rs"]
        return {"schema_version": 1, "purpose": "supplementary_source_classification_not_a_coverage_gate", "qualified_for_coverage_exclusions": False, "macro_expansion_resolved": False, "root": snapshot["crate_root"], "files": [{"path": "src/lib.rs", "sha256": entry["sha256"], "byte_length": entry["bytes"], "line_count": entry["lines"]}]}

    def test_classifier_binding_requires_exact_source_bytes_and_root(self):
        """Matching hashes bind observations without making syntax a coverage gate."""

        snapshot = P.capture_sources(self.root)
        report = self.classifier_report(snapshot)
        P.validate_classification_sources(snapshot, report)
        mutations = (("sha256", "0" * 64), ("byte_length", 0), ("line_count", 1000), ("line_count", True), ("path", "src/missing.rs"), ("path", "src/../lib.rs"))
        for key, value in mutations:
            damaged = copy.deepcopy(report)
            damaged["files"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(P.CoverageProvenanceError):
                P.validate_classification_sources(snapshot, damaged)
        for key, value in (("root", "/elsewhere"), ("schema_version", True), ("purpose", "new"), ("qualified_for_coverage_exclusions", True), ("macro_expansion_resolved", True), ("files", []), ("files", [None]), ("files", report["files"] * 2)):
            damaged = copy.deepcopy(report)
            damaged[key] = value
            with self.subTest(key=key), self.assertRaises(P.CoverageProvenanceError):
                P.validate_classification_sources(snapshot, damaged)
        with self.assertRaises(P.CoverageProvenanceError):
            P.validate_classification_sources(snapshot, None)


if __name__ == "__main__":
    unittest.main()
