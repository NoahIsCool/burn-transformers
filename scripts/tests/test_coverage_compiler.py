"""Pure artifact-message tests and opt-in tiny native compiler-wrapper exercises."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "coverage_compiler.py"
SPEC = importlib.util.spec_from_file_location("coverage_compiler", SCRIPT)
COMPILER = importlib.util.module_from_spec(SPEC)
with mock.patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    SPEC.loader.exec_module(COMPILER)


def artifact(path, emit="dep-info"):
    """Construct only the native protocol fields consumed by the wrapper."""

    return json.dumps({"$message_type": "artifact", "artifact": path, "emit": emit}).encode() + b"\n"


class ArtifactMessageTests(unittest.TestCase):
    """Distinguish native messages from diagnostics and validate file containment."""

    def test_native_paths_are_not_split_or_make_unescaped(self):
        """Artifact JSON makes spaces, hashes and dollar signs unambiguous."""

        for path in ("/tmp/output with space/雪.d", "hash#file.d", "money$file.d", "back\\slash.d"):
            self.assertEqual(COMPILER.artifact_message(artifact(path)), {"artifact": path, "emit": "dep-info"})
        self.assertIsNone(COMPILER.artifact_message(b"ordinary native stderr\n"))
        self.assertIsNone(COMPILER.artifact_message(b'{"$message_type":"diagnostic","level":"warning"}\n'))
        self.assertIsNone(COMPILER.artifact_message(b"{}\n"))

    def test_malformed_or_unsupported_messages_fail_closed(self):
        """Ambiguous keys, lossy JSON and unknown emission kinds cannot be records."""

        for line in (b"{", b'{"artifact":"\xff"}', b'{"a":1,"a":2}',
                     b'{"a":NaN}', artifact(""), artifact(None), artifact("a\0.d"),
                     artifact("a.d", "future-kind"), artifact("a.d", None), "wrong type"):
            with self.subTest(line=line), self.assertRaises(COMPILER.CompilerCaptureError):
                COMPILER.artifact_message(line)
        with mock.patch.object(COMPILER, "MAX_MESSAGE_BYTES", 1), self.assertRaises(COMPILER.CompilerCaptureError):
            COMPILER.artifact_message(b"ab")
        nested = b'{"nested":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}"
        with self.assertRaises(COMPILER.CompilerCaptureError):
            COMPILER.artifact_message(nested)

    def test_identities_require_dep_info_and_real_compiled_artifacts(self):
        """Files are byte-bound only inside the designated output directory."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "fixture.d").write_bytes(b"out: source.rs\nsource.rs:\n")
            (root / "libfixture.rmeta").write_bytes(b"metadata")
            messages = [COMPILER.artifact_message(artifact("fixture.d")),
                        COMPILER.artifact_message(artifact(str(root / "libfixture.rmeta"), "metadata"))]
            records = COMPILER.artifact_identities(messages, root, root)
            self.assertEqual(len(records), 2)
            for row in records:
                self.assertEqual(row["identity"]["sha256"], hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest())
            for invalid in ([], messages[:1], messages[1:], messages * 2):
                with self.subTest(invalid=invalid), self.assertRaises(COMPILER.CompilerCaptureError):
                    COMPILER.artifact_identities(invalid, root, root)

    def test_outside_paths_symlinks_missing_and_nonregular_outputs_fail(self):
        """Native-looking paths cannot make the wrapper read unrelated files."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            (root / "outside.d").write_bytes(b"private")
            (outputs / "alias.d").symlink_to(root / "outside.d")
            for path in (root / "outside.d", outputs / "alias.d", outputs / "missing.d", outputs):
                with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                    COMPILER.artifact_identities([COMPILER.artifact_message(artifact(str(path)))], outputs, outputs)
            (root / "alias-directory").symlink_to(outputs, target_is_directory=True)
            with self.assertRaises(COMPILER.CompilerCaptureError):
                COMPILER.checked_directory(root / "alias-directory")

    def test_records_never_overwrite_and_probes_do_not_need_capture_roots(self):
        """A repeated invocation identity is an error; Cargo probes remain transparent."""

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.json"
            COMPILER.write_record(path, {"capture_valid": False})
            with self.assertRaises(FileExistsError):
                COMPILER.write_record(path, {"capture_valid": True})
            self.assertFalse(json.loads(path.read_text())["capture_valid"])
        self.assertTrue(COMPILER.emission_requested(["--emit=dep-info,link"]))
        self.assertTrue(COMPILER.emission_requested(["--emit", "metadata"]))
        self.assertFalse(COMPILER.emission_requested(["--print=cfg"]))
        with mock.patch.object(COMPILER.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)):
            self.assertEqual(COMPILER.capture(["rustc", "-Vv"], {}, Path.cwd()), 7)
        with self.assertRaises(COMPILER.CompilerCaptureError):
            COMPILER.capture([], {}, Path.cwd())

    def test_default_emission_response_files_and_linking_prints_are_not_probes(self):
        """Unsupported compiler forms fail explicitly instead of losing evidence."""

        for arguments in (["input.rs"], ["@args"], ["--print=link-args", "input.rs"],
                          ["--print"], ["--print=cfg,unknown"], ["--emit=link", "--print=cfg"], ["-C", "help", "input.rs"]):
            self.assertFalse(COMPILER.compiler_probe(arguments))
        for arguments in (["-Vv"], ["--version"], ["--print", "cfg"], ["-C", "help"],
                          ["-", "--crate-name", "___", "--print=file-names", "--crate-type", "rlib", "--print=cfg", "--print=split-debuginfo"]):
            self.assertTrue(COMPILER.compiler_probe(arguments))
        for arguments in (["rustc", "input.rs"], ["rustc", "@args"], ["rustc", "--emit=link", "@args"]):
            with mock.patch.object(COMPILER.subprocess, "run") as run, self.assertRaises(COMPILER.CompilerCaptureError):
                COMPILER.capture(arguments, {}, Path.cwd())
            run.assert_not_called()

    def test_version_probe_bounds_diagnostics_encoding_and_native_failure(self):
        """Spooling version text avoids unlimited memory buffering of probe output."""

        for data, errors, code, valid in ((b"rustc fixture\n", b"", 0, True),
                (b"rustc " + b"x" * COMPILER.MAX_VERSION_BYTES, b"", 0, False),
                (b"rustc fixture\n", b"warning", 0, False), (b"rustc fixture\n", b"", 7, False),
                (b"not rustc", b"", 0, False), (b"rustc \xff", b"", 0, False)):
            def probe(command, **kwargs):
                """Write into the wrapper's real unlinked temporary streams."""

                kwargs["stdout"].write(data)
                kwargs["stderr"].write(errors)
                return subprocess.CompletedProcess(command, code)

            with self.subTest(valid=valid, code=code), mock.patch.object(COMPILER.subprocess, "run", side_effect=probe):
                if valid:
                    self.assertEqual(COMPILER.compiler_identity("rustc", Path.cwd(), {}), data.decode())
                else:
                    with self.assertRaises(COMPILER.CompilerCaptureError):
                        COMPILER.compiler_identity("rustc", Path.cwd(), {})


class CaptureLifecycleTests(unittest.TestCase):
    """Drive the real lifecycle around controlled native stderr/status boundaries."""

    def run_capture(self, native_output, native_code=0, mutation=None):
        """Retain child bytes and records without a real compiler or GPU."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            capture = root / "capture"
            capture.mkdir()
            (root / "fixture.d").write_bytes(b"out: src.rs\nsrc.rs:\n")
            (root / "fixture.rmeta").write_bytes(b"metadata")
            child = mock.MagicMock()
            child.__enter__.return_value = child
            child.stderr = io.BytesIO(native_output)
            child.wait.return_value = native_code
            output = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", write_through=True)
            original_fingerprint = COMPILER.fingerprint_file

            def observed(path, **kwargs):
                """Optionally mutate source storage at a selected validation boundary."""

                if mutation is not None:
                    mutation(path)
                return original_fingerprint(path, **kwargs)

            with (mock.patch.object(COMPILER, "compiler_identity", return_value="rustc fixture\n"),
                  mock.patch.object(COMPILER.subprocess, "Popen", return_value=child),
                  mock.patch.object(COMPILER.sys, "stderr", output),
                  mock.patch.object(COMPILER, "fingerprint_file", side_effect=observed)):
                code = COMPILER.capture(["rustc", "--emit=dep-info,metadata", "src.rs"],
                    {"TFM_COVERAGE_CAPTURE_ROOT": str(capture), "TFM_COVERAGE_ARTIFACT_ROOT": str(root)}, root)
            record_path, = capture.glob("*.json")
            record = json.loads(record_path.read_text())
            forwarded = output.buffer.getvalue()
            output.close()
            return code, record, forwarded

    def test_malformed_deep_and_oversized_messages_are_drained_before_failure(self):
        """A parse failure never strands the child behind a full stderr pipe."""

        for malformed in (b"{\n", b'{"a":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}\n",
                          b"x" * (COMPILER.MAX_MESSAGE_BYTES + 1) + b"\n"):
            native = malformed + b"later stderr must still be forwarded\n" + artifact("fixture.d") + artifact("fixture.rmeta", "metadata")
            code, record, forwarded = self.run_capture(native)
            self.assertEqual((code, record["process_exit_code"]), (1, 0))
            self.assertFalse(record["capture_valid"])
            self.assertNotIn("artifacts", record)
            self.assertTrue(forwarded.startswith(native))

    def test_native_failure_never_adopts_plausible_artifact_messages(self):
        """Native status wins even if an earlier compilation stage emitted files."""

        native = artifact("fixture.d") + artifact("fixture.rmeta", "metadata")
        code, record, forwarded = self.run_capture(native, native_code=7)
        self.assertEqual((code, record["process_exit_code"]), (7, 7))
        self.assertFalse(record["capture_valid"])
        self.assertNotIn("artifacts", record)
        self.assertEqual(forwarded, native)

    def test_stable_success_is_byte_bound_and_missing_artifacts_are_failed(self):
        """A zero exit needs both validated dependency and compiled outputs."""

        native = artifact("fixture.d") + artifact("fixture.rmeta", "metadata")
        code, record, forwarded = self.run_capture(native)
        self.assertEqual(code, 0)
        self.assertTrue(record["capture_valid"])
        self.assertEqual(forwarded, native)
        self.assertFalse(record["compiler_dependency_closure_verified"])
        for invalid in (b"", artifact("missing.d") + artifact("fixture.rmeta", "metadata"), native * 9):
            code, record, _ = self.run_capture(invalid)
            self.assertEqual(code, 1)
            self.assertFalse(record["capture_valid"])

    def test_artifact_changes_before_copy_and_before_terminal_validation_fail(self):
        """Rechecking only the copied dep-info would miss changed compiled binaries."""

        native = artifact("fixture.d") + artifact("fixture.rmeta", "metadata")
        for filename in ("fixture.d", "fixture.rmeta"):
            calls = 0

            def mutate(path):
                """Change same-length bytes on the second hash of one original file."""

                nonlocal calls
                if Path(path).name == filename:
                    calls += 1
                    if calls == 2:
                        Path(path).write_bytes(Path(path).read_bytes().replace(b"s", b"z").replace(b"m", b"n"))

            with self.subTest(filename=filename):
                code, record, _ = self.run_capture(native, mutation=mutate)
                self.assertEqual(code, 1)
                self.assertFalse(record["capture_valid"])
                self.assertIn("changed", record["capture_error"])


@unittest.skipUnless(os.environ.get("TFM_COMPILER_CAPTURE_NATIVE_TESTS") == "1",
                     "set TFM_COMPILER_CAPTURE_NATIVE_TESTS=1 for isolated rustc captures")
class NativeCompilerCaptureTests(unittest.TestCase):
    """Exercise real rustc artifact emission, not a simulated command transcript."""

    def test_success_failure_and_probe_capture_contract(self):
        """Capture both native producers with exact argv and separately retained dep-info."""

        rustup = shutil.which("rustup")
        self.assertIsNotNone(rustup)
        for toolchain in ("stable", "nightly"):
            with self.subTest(toolchain=toolchain), tempfile.TemporaryDirectory(prefix="tfm compiler ") as temporary:
                root = Path(temporary).resolve()
                output = root / "compiler outputs"
                output.mkdir()
                capture = root / "capture"
                capture.mkdir()
                source = root / "entry point.rs"
                source.write_text("//! Tiny compiler capture fixture.\n/// An exported constant.\npub const VALUE: u32 = 7;\n")
                compiler = subprocess.check_output([rustup, "which", "--toolchain", toolchain, "rustc"], text=True).strip()
                env = {**os.environ, "TFM_COVERAGE_CAPTURE_ROOT": str(capture),
                       "TFM_COVERAGE_ARTIFACT_ROOT": str(output)}
                arguments = [compiler, "--edition=2024", "--crate-name", "capture_fixture",
                             "--crate-type", "lib", str(source), "--out-dir", str(output),
                             "--emit=dep-info,metadata", "--error-format=json", "--json=artifacts"]
                result = subprocess.run([sys.executable, str(SCRIPT), *arguments], cwd=root, env=env,
                                        capture_output=True, check=False, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(result.stdout, b"")
                self.assertEqual(len(result.stderr.splitlines()), 2)
                record_path, = capture.glob("*.json")
                record = json.loads(record_path.read_text())
                self.assertTrue(record["capture_valid"])
                self.assertEqual(record["command"], arguments)
                self.assertEqual(record["process_exit_code"], 0)
                self.assertFalse(record["production_only_exclusions_qualified"])
                dep, = [row for row in record["artifacts"] if row["emit"] == "dep-info"]
                self.assertEqual((capture / dep["retained_dep_info"]).read_bytes(), Path(dep["path"]).read_bytes())
                probe = subprocess.run([sys.executable, str(SCRIPT), compiler, "-Vv"], cwd=root, env=env,
                                       capture_output=True, check=False, timeout=30)
                self.assertEqual(probe.returncode, 0)
                self.assertEqual(probe.stderr, b"")
                self.assertEqual(len(list(capture.glob("*.json"))), 1)
                source.write_text("this is not Rust\n")
                failed = subprocess.run([sys.executable, str(SCRIPT), *arguments], cwd=root, env=env,
                                        capture_output=True, check=False, timeout=30)
                self.assertNotEqual(failed.returncode, 0)
                failures = [json.loads(path.read_text()) for path in capture.glob("*.json") if path != record_path]
                failure, = failures
                self.assertEqual(failure["process_exit_code"], failed.returncode)
                self.assertFalse(failure["capture_valid"])
                self.assertNotIn("artifacts", failure)

    def test_cargo_workspace_wrapper_captures_real_library_and_unit_artifacts(self):
        """Native Cargo probes and compiler-artifact messages work without Burn builds."""

        with tempfile.TemporaryDirectory(prefix="tfm cargo capture ") as temporary:
            root = Path(temporary).resolve()
            (root / "src").mkdir()
            output = root / "target"
            output.mkdir()
            capture = root / "capture"
            capture.mkdir()
            (root / "Cargo.toml").write_text('[package]\nname="capture-fixture"\nversion="0.1.0"\nedition="2024"\n[workspace]\n')
            (root / "src/lib.rs").write_text('//! Capture fixture.\n/// An exported constant.\npub const VALUE:u32=7;\n#[cfg(test)] mod tests;\n')
            (root / "src/tests.rs").write_text('/// Test-only input.\n#[test] fn value(){assert_eq!(super::VALUE,7);}\n')
            env = {**os.environ, "RUSTC_WORKSPACE_WRAPPER": str(SCRIPT),
                   "TFM_COVERAGE_CAPTURE_ROOT": str(capture), "TFM_COVERAGE_ARTIFACT_ROOT": str(output),
                   "CARGO_INCREMENTAL": "0", "CARGO_TARGET_DIR": str(output)}
            # A caller's unrelated cache wrapper must not change this fixture's
            # compiler command or suppress the native compilation under test.
            env.pop("RUSTC_WRAPPER", None)
            env.pop("RUSTC", None)
            for operation in ("build", "test"):
                result = subprocess.run(["cargo", "+nightly", operation, "--manifest-path", str(root / "Cargo.toml"),
                                         "--offline", "--lib", "--message-format=json"], cwd=root, env=env,
                                        capture_output=True, check=False, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
            records = [json.loads(path.read_text()) for path in capture.glob("*.json")]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["capture_valid"] for record in records))
            production, = [record for record in records if "--test" not in record["command"]]
            unit, = [record for record in records if "--test" in record["command"]]
            for record, test_input in ((production, False), (unit, True)):
                dep, = [row for row in record["artifacts"] if row["emit"] == "dep-info"]
                data = (capture / dep["retained_dep_info"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), dep["identity"]["sha256"])
                self.assertEqual(b"src/tests.rs" in data, test_input)
                self.assertTrue(any(row["emit"] == "link" for row in record["artifacts"]))


if __name__ == "__main__":
    unittest.main()
