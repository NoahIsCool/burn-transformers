"""Fail-closed native artifact binding, plus opt-in tiny Cargo cache differentials."""

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "coverage_binding.py"
SPEC = importlib.util.spec_from_file_location("coverage_binding", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BIND = importlib.util.module_from_spec(SPEC)
with mock.patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
    SPEC.loader.exec_module(BIND)
    from coverage_provenance import capture_sources, require_same_sources


VERSION = "rustc 1.94.0-nightly (4f14395c3 2025-12-31)\nbinary: rustc\ncommit-hash: 4f14395c37db4c1be874e6b0ace6721674223c22\ncommit-date: 2025-12-31\nhost: aarch64-apple-darwin\nrelease: 1.94.0-nightly\nLLVM version: 21.1.8\n"


def put(path: Path, contents: str) -> dict:
    """Create a small fixture and return its actual byte identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return BIND.fingerprint_file(path)


def native_crate(crate: Path) -> None:
    """Create an offline dependency-free crate with an external, spaced test input."""

    put(crate / "Cargo.toml", '[package]\nname="binding_fixture"\nversion="0.1.0"\nedition="2024"\n[workspace]\n[features]\nfixture=[]\n')
    put(crate / "Cargo.lock", 'version = 4\n[[package]]\nname = "binding_fixture"\nversion = "0.1.0"\n')
    put(crate / "src/lib.rs", '#[path = "tests/production.rs"]\nmod production;\npub fn value() -> u32 { production::value() }\n#[cfg(test)]\n#[path = "unit tests.rs"]\nmod tests;\n')
    put(crate / "src/tests/production.rs", 'pub fn value() -> u32 { 7 }\n')
    put(crate / "src/unit tests.rs", '#[test]\nfn value() { assert_eq!(super::value(), 7); }\n')
    put(crate / "tests/integration.rs", '#[test]\nfn value() { assert_eq!(binding_fixture::value(), 7); }\n')


def binding_fixture(crate: Path) -> tuple[dict, dict]:
    """Build authenticated synthetic compiler/export observations, not coverage math."""

    native_crate(crate)
    artifact_root, root, export = crate / "target/llvm-cov-target", crate / "target/records", crate / "target/export"
    artifact_root.mkdir(parents=True)
    root.mkdir()
    export.mkdir()
    tool = crate / "target/llvm-cov"
    tool_identity = put(tool, "native tool fixture")
    rustc = crate / "target/rustc"
    rustc_identity = put(rustc, "native compiler fixture")
    production = artifact_root / "libbinding_fixture.rlib"
    targets = [{"name": "binding_fixture", "source": str(crate / "src/lib.rs"), "test": False},
               {"name": "binding_fixture", "source": str(crate / "src/lib.rs"), "test": True},
               {"name": "integration", "source": str(crate / "tests/integration.rs"), "test": True}]
    setup = {"compiler_root": str(root), "artifact_root": str(artifact_root), "export_root": str(export),
             "wrappers": {}, "tools": {str(tool): tool_identity, str(rustc): rustc_identity}, "targets": targets,
             "rustc": str(rustc), "llvm_cov": str(tool), "compiler_version": VERSION}
    links = []
    for index, target in enumerate(targets):
        link = production if index == 0 else artifact_root / f"test-{index}"
        identity = put(link, f"compiled object {index}")
        inputs = [str(Path(target["source"]).relative_to(crate))]
        if index == 1:
            inputs.append("src/unit tests.rs")
            links.append({"path": str(link), "identity": identity})
        elif index == 2:
            links.append({"path": str(link), "identity": identity})
        escaped = [value.replace(" ", "\\ ") for value in inputs]
        dependency = "out: " + " ".join(escaped) + "\n\n" + "".join(value + ":\n" for value in escaped)
        dep_path = artifact_root / f"rustc-{index}.d"
        dep_identity = put(dep_path, dependency)
        put(root / f"rustc-{index}.d", dependency)
        arguments = [str(rustc), "--crate-name", target["name"], target["source"], "-C", "instrument-coverage", "--cfg", 'feature="fixture"']
        if target["test"]:
            arguments.append("--test")
        if index == 2:
            arguments += ["--extern", f"binding_fixture={production}"]
        record = {"schema_version": 1, "capture_valid": True, "process_exit_code": 0, "command": arguments,
                  "cwd": str(crate), "compiler_version": VERSION, "artifacts": [
                      {"path": str(dep_path), "identity": dep_identity, "emit": "dep-info", "retained_dep_info": f"rustc-{index}.d"},
                      {"path": str(link), "identity": identity, "emit": "link"}]}
        BIND.write_json(root / f"rustc-{index}.json", record)
    profile = artifact_root / "fixture.profdata"
    profile_identity = put(profile, "merged profile")
    report = export / "native-coverage.json"
    report_identity = put(report, json.dumps({"type": "llvm.coverage.json.export", "version": "3.0.1", "data": [
        {"files": [{"filename": str(crate / "src/lib.rs")}]}]}))
    argv = [str(tool), "export", "-format=text", f"-instr-profile={profile}"]
    for link in links:
        argv += ["-object", link["path"]]
    BIND.write_json(export / "export.json", {"schema_version": 1, "capture_valid": True, "process_exit_code": 0,
        "command": argv, "cwd": str(crate), "tool_identity": tool_identity,
        "inputs": [{"path": str(profile), "identity": profile_identity}, *links],
        "report": {"path": str(report), "identity": report_identity}})
    return setup, capture_sources(crate)


class BindingTests(unittest.TestCase):
    """Distinct malformed contracts must fail rather than waive missing evidence."""

    def test_valid_capture_has_narrow_claims(self):
        """Bind production+unit+integration inputs without authorizing exclusions."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory).resolve()
            setup, snapshot = binding_fixture(crate)
            result = BIND.validate_capture(setup, snapshot, crate)
            self.assertTrue(result["artifact_binding_verified"])
            self.assertTrue(result["compiler_reported_inputs_verified"])
            self.assertFalse(result["compiler_dependency_closure_verified"])
            self.assertFalse(result["production_only_exclusions_qualified"])
            self.assertEqual(len(result["targets"]), 3)

    def test_export_grammar_rejects_implicit_or_duplicate_inputs(self):
        """Response files, summary-only output and path aliases cannot hide objects."""

        prefix = ["export", "-format=text", "-instr-profile=/tmp/p", "-object", "/tmp/o"]
        self.assertEqual(BIND.export_inputs(prefix), ("/tmp/p", ["/tmp/o"]))
        for argv in (["show"], prefix + ["-summary-only"], prefix + ["@args"], prefix + ["-object", "/tmp/o"],
                     prefix + ["-instr-profile=/tmp/q"], prefix + ["-object"], prefix[:3],
                     prefix + ["-ignore-filename-regex", "a", "-ignore-filename-regex", "b"]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                BIND.export_inputs(argv)

    def test_producer_requires_exact_nightly_and_unique_fields(self):
        """A mutable nightly alias or stable LLVM identity does not imply compatibility."""

        self.assertEqual(BIND.producer_from_version(VERSION), BIND.COVERAGE_PRODUCER)
        for version in (VERSION.replace("21.1.8", "21.1.9"), VERSION + "release: 1.94.0-nightly\n", "rustc x\n"):
            with self.subTest(version=version), self.assertRaises((ValueError, KeyError)):
                BIND.producer_from_version(version)

    def test_literal_compiler_arguments_are_unambiguous(self):
        """Reject response/remapping and disabled coverage instead of guessing flags."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory)
            native_crate(crate)
            prefix = ["rustc", "--crate-name=fixture", "src/lib.rs", "-C", "instrument-coverage"]
            self.assertEqual(BIND.compiler_arguments(prefix, crate)["name"], "fixture")
            for command in (prefix + ["@args"], prefix + ["--remap-path-prefix=a=b"], prefix + ["--test", "--test"],
                            prefix + ["src/lib.rs"], prefix + ["--crate-name=x"], prefix[:-1] + ["instrument-coverage=off"]):
                with self.subTest(command=command), self.assertRaises(ValueError):
                    BIND.compiler_arguments(command, crate)

    def test_changed_inputs_and_incomplete_captures_fail_closed(self):
        """Exercise stale binary/source/profile, producer, target and link accounting."""

        cases = ("object", "profile", "source", "dep_copy", "missing_target", "extra_file", "failed_record",
                 "different_cfg", "wrong_extern", "same_source_without_extern", "missing_dep_source", "stale_export_object",
                 "unexpected_filter", "tool", "raw_report")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                crate = Path(directory).resolve()
                setup, snapshot = binding_fixture(crate)
                root, export, artifacts = Path(setup["compiler_root"]), Path(setup["export_root"]), Path(setup["artifact_root"])
                if case in ("object", "profile", "source", "dep_copy", "tool", "raw_report"):
                    path = {"object": artifacts / "test-1", "profile": artifacts / "fixture.profdata", "source": crate / "src/lib.rs",
                            "dep_copy": root / "rustc-1.d", "tool": crate / "target/llvm-cov", "raw_report": export / "native-coverage.json"}[case]
                    path.write_text("changed")
                elif case == "missing_target":
                    (root / "rustc-1.json").unlink()
                elif case == "extra_file":
                    (root / "unexplained").write_text("unknown")
                elif case in ("stale_export_object", "unexpected_filter"):
                    path = export / "export.json"
                    record = json.loads(path.read_text())
                    if case == "stale_export_object":
                        record["command"][-1] = str(artifacts / "libbinding_fixture.rlib")
                    elif case == "unexpected_filter":
                        record["command"] += ["-ignore-filename-regex", "src/.*"]
                    path.write_text(json.dumps(record))
                else:
                    path = root / "rustc-2.json"
                    record = json.loads(path.read_text())
                    if case == "failed_record":
                        record["capture_valid"] = False
                    elif case == "different_cfg":
                        record["command"] += ["--cfg", "unexpected"]
                    elif case == "wrong_extern":
                        record["command"][-1] = f"binding_fixture={artifacts / 'test-1'}"
                    elif case == "missing_dep_source":
                        data = "out: src/lib.rs\n\nsrc/lib.rs:\n"
                        identity = put(artifacts / "rustc-2.d", data)
                        put(root / "rustc-2.d", data)
                        record["artifacts"][0]["identity"] = identity
                    elif case == "same_source_without_extern":
                        setup["targets"][2]["source"] = str(crate / "src/lib.rs")
                        record["command"][3] = str(crate / "src/lib.rs")
                        record["command"] = record["command"][:-2]
                        data = "out: src/lib.rs\n\nsrc/lib.rs:\n"
                        identity = put(artifacts / "rustc-2.d", data)
                        put(root / "rustc-2.d", data)
                        record["artifacts"][0]["identity"] = identity
                    path.write_text(json.dumps(record))
                with self.assertRaises((ValueError, OSError)):
                    BIND.validate_capture(setup, snapshot, crate)

    def test_json_and_path_bounds(self):
        """Malformed records and symlink substitutions are not authentic observations."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "record.json"
            for text in ('{"a":1,"a":2}', '{"a":NaN}', '[]', '{' * 2000):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    BIND.read_json(path)
            path.write_text('{}')
            with self.assertRaises(ValueError):
                BIND.read_json(path, 1)
            link = root / "alias.json"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                BIND.contained_file(str(link), root)

    def test_metadata_queries_reject_diagnostics_and_excess_output(self):
        """Preflight cannot bypass warning policy or buffer unbounded query text."""

        for data, diagnostic, code, valid in ((b"version\n", b"", 0, True), (b"x", b"warning", 0, False),
                                             (b"x", b"", 1, False), (b"\xff", b"", 0, False), (b"x" * 33, b"", 0, False)):
            def query(argv, **kwargs):
                """Populate the tool's real bounded temporary output streams."""

                kwargs["stdout"].write(data)
                kwargs["stderr"].write(diagnostic)
                return subprocess.CompletedProcess(argv, code)

            with self.subTest(valid=valid, code=code), mock.patch.object(BIND.subprocess, "run", side_effect=query), mock.patch.object(BIND, "MAX_RECORD_BYTES", 32):
                if valid:
                    self.assertEqual(BIND.query(["tool"], Path.cwd(), {}), data.decode())
                else:
                    with self.assertRaises(ValueError):
                        BIND.query(["tool"], Path.cwd(), {})

    def test_export_capture_retains_failures_and_terminates_oversized_output(self):
        """Zero status cannot excuse changed inputs; failed child/output stays failed."""

        for scenario in ("success", "native_failure", "changed_input", "oversized"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                crate = Path(directory).resolve()
                artifact_root, export = crate / "artifacts", crate / "export"
                artifact_root.mkdir()
                export.mkdir()
                tool, profile, obj = crate / "llvm-cov", artifact_root / "p", artifact_root / "o"
                for path in (tool, profile, obj):
                    put(path, "original")
                child = mock.MagicMock()
                child.stdout = io.BytesIO(b'{"export":true}')
                child.wait.return_value = 7 if scenario == "native_failure" else 0

                def enter():
                    """Model an input mutation during the native execution interval."""

                    if scenario == "changed_input":
                        obj.write_text("mutated")
                    return child

                child.__enter__.side_effect = enter
                stdout = mock.Mock(buffer=io.BytesIO())
                with mock.patch.object(BIND.subprocess, "Popen", return_value=child), mock.patch.object(BIND.sys, "stdout", stdout), \
                     mock.patch.object(BIND.sys, "stderr", io.StringIO()), mock.patch.object(BIND, "MAX_EXPORT_BYTES", 4 if scenario == "oversized" else 1024):
                    code = BIND.capture_export(tool, export, artifact_root, ["export", "-format=text", f"-instr-profile={profile}", "-object", str(obj)])
                record = json.loads((export / "export.json").read_text())
                self.assertEqual(record["capture_valid"], scenario == "success")
                self.assertEqual(code, 0 if scenario == "success" else 7 if scenario == "native_failure" else 1)
                if scenario == "oversized":
                    child.kill.assert_called_once()
                else:
                    self.assertEqual(record["process_exit_code"], 7 if scenario == "native_failure" else 0)
                if scenario in ("success", "native_failure"):
                    self.assertEqual(stdout.buffer.getvalue(), b'{"export":true}')

    def test_expected_targets_respect_feature_requirements_and_standalone_scope(self):
        """Missing production/integration contexts and workspace build scripts fail."""

        with tempfile.TemporaryDirectory() as directory:
            crate = Path(directory).resolve()
            native_crate(crate)
            library = {"kind": ["lib"], "name": "binding_fixture", "src_path": str(crate / "src/lib.rs")}
            integration = {"kind": ["test"], "name": "integration", "src_path": str(crate / "tests/integration.rs"), "required-features": ["fixture"]}
            package = {"id": "fixture", "manifest_path": str(crate / "Cargo.toml"), "targets": [library, integration]}
            metadata = {"packages": [package], "workspace_members": ["fixture"]}
            self.assertEqual(len(BIND.expected_targets(metadata, crate, {"fixture"})), 3)
            with self.assertRaises(ValueError):
                BIND.expected_targets(metadata, crate, set())
            package["targets"].append({"kind": ["custom-build"], "name": "build"})
            with self.assertRaises(ValueError):
                BIND.expected_targets(metadata, crate, {"fixture"})
            metadata["workspace_members"].append("other")
            with self.assertRaises(ValueError):
                BIND.expected_targets(metadata, crate, {"fixture"})

    def test_inherited_overrides_fail_before_queries_or_files(self):
        """An explicit capture run cannot silently replace an existing build wrapper."""

        for variable in ("RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS", "LLVM_COV", "LLVM_PROFILE_FILE", "CARGO_LLVM_COV_TARGET_DIR"):
            with self.subTest(variable=variable), mock.patch.object(BIND, "query") as query, self.assertRaises(ValueError):
                BIND.prepare_capture(Path("/missing"), Path("/missing"), "nightly", "fixture", {variable: "override"})
            query.assert_not_called()

@unittest.skipUnless(os.environ.get("TFM_COVERAGE_BINDING_NATIVE_TESTS") == "1", "opt-in tiny offline native Cargo/LLVM fixture")
class NativeBindingTests(unittest.TestCase):
    """Differentially prove the installed overrides and fresh workspace cache keys."""

    def test_cache_reuse_then_unique_wrapper_and_exact_export(self):
        """Same wrapper reuses artifacts; a new path recompiles all selected targets."""

        with tempfile.TemporaryDirectory(prefix="tfm coverage binding ") as directory:
            crate = Path(directory).resolve()
            native_crate(crate)
            (crate / "target").mkdir()
            environment = {key: value for key, value in os.environ.items() if not key.startswith("TFM_")}
            environment["CARGO_INCREMENTAL"] = "0"
            environment["CARGO_TERM_COLOR"] = "never"
            first, second, final = (crate / "target" / name for name in ("first", "second", "final"))
            for path in (first, second, final):
                path.mkdir()
            command = ["cargo", "+nightly", "test", "--locked", "--offline", "--no-default-features", "--features", "fixture",
                       "--lib", "--tests", "--no-run", "--target-dir", str(crate / "target/llvm-cov-target")]

            def run(argv, env):
                """Require warning-free native completion, retaining failure diagnostics."""

                result = subprocess.run(argv, cwd=crate, env=env, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                stderr = result.stderr
                if "--disable-default-ignore-filename-regex" in argv:
                    notice = "warning: --disable-default-ignore-filename-regex option is unstable\n"
                    self.assertEqual(stderr.count(notice), 1, stderr)
                    stderr = stderr.replace(notice, "", 1)
                self.assertNotIn("warning:", stderr.lower(), result.stderr)
                self.assertNotIn("mismatched", result.stderr.lower(), result.stderr)
                return result

            env1, setup1 = BIND.prepare_capture(crate, first, "nightly", "fixture", environment)
            env1["RUSTFLAGS"] = "-C instrument-coverage"
            run(command, env1)
            initial = sorted(Path(setup1["compiler_root"]).glob("*.json"))
            self.assertEqual(len(initial), 3)
            run(command, env1)
            self.assertEqual(initial, sorted(Path(setup1["compiler_root"]).glob("*.json")))
            env2, setup2 = BIND.prepare_capture(crate, second, "nightly", "fixture", environment)
            env2["RUSTFLAGS"] = "-C instrument-coverage"
            run(command, env2)
            self.assertEqual(len(list(Path(setup2["compiler_root"]).glob("*.json"))), 3)
            env3, setup3 = BIND.prepare_capture(crate, final, "nightly", "fixture", environment)
            snapshot = capture_sources(crate)
            run(["cargo", "+nightly", "llvm-cov", "--locked", "--offline", "--no-default-features", "--features", "fixture",
                 "--lib", "--tests", "--json", "--output-path", str(final / "coverage.json"), "--disable-default-ignore-filename-regex"], env3)
            require_same_sources(snapshot, capture_sources(crate))
            result = BIND.validate_capture(setup3, snapshot, crate)
            self.assertTrue(result["artifact_binding_verified"])
            self.assertTrue(result["compiler_reported_inputs_verified"])
            self.assertTrue(result["no_export_filter_verified"])
            self.assertEqual(len(result["llvm_objects"]), 2)
            self.assertFalse(result["production_only_exclusions_qualified"])
            raw = json.loads(Path(result["native_report"]["path"]).read_text())
            rewritten = json.loads((final / "coverage.json").read_text())
            self.assertNotIn("cargo_llvm_cov", raw)
            self.assertEqual(rewritten["cargo_llvm_cov"]["version"], "0.6.21")
            self.assertIn(str(crate / "src/tests/production.rs"), {file["filename"] for block in raw["data"] for file in block["files"]})
            proof_path = os.environ.get("TFM_COVERAGE_BINDING_PROOF")
            if proof_path:
                BIND.write_json(Path(proof_path), {
                    "schema_version": 1, "purpose": "tiny_native_regression_not_release_or_exclusion_evidence",
                    "fresh_workspace_capture_counts": [3, 0, 3], "setup": setup3, "sources": snapshot,
                    "compiler_records": [BIND.read_json(path) for path in sorted(Path(setup3["compiler_root"]).glob("*.json"))],
                    "native_export": BIND.read_json(Path(setup3["export_root"]) / "export.json"), "binding": result,
                    "reviewed_option_notice_count": 1, "other_native_warning_count": 0,
                })

    def test_real_runner_requires_the_bound_native_report(self):
        """Exercise the actual opt-in runner end to end without the Burn dependency graph."""

        spec = importlib.util.spec_from_file_location("binding_native_runner", SCRIPT.with_name("check_release.py"))
        runner = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(SCRIPT.parent), *sys.path]):
            spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory(prefix="tfm native runner ") as directory:
            crate = Path(directory).resolve()
            native_crate(crate)
            with mock.patch.object(runner, "CRATE", crate), mock.patch.object(runner, "FEATURES", "fixture"), mock.patch.object(sys, "argv", [
                "check_release", "--gate", "coverage", "--toolchain", "nightly", "--offline", "--coverage-compiler-evidence",
            ]), mock.patch("builtins.print"):
                code = runner.main()
            result_path = next((crate / "target/release-gates").glob("*/results.json"))
            result = json.loads(result_path.read_text())
            self.assertEqual(code, 0, json.dumps(result))
            self.assertTrue(result["passed"])
            row = result["gates"][0]
            self.assertTrue(row["compiler_evidence"]["artifact_binding_verified"])
            self.assertTrue(row["compiler_evidence"]["compiler_reported_inputs_verified"])
            self.assertFalse(row["compiler_evidence"]["production_only_exclusions_qualified"])
            binding = json.loads((result_path.parent / row["compiler_evidence"]["binding"]).read_text())
            self.assertTrue(binding["native_aggregate"]["passed"])
            self.assertEqual(binding["reviewed_tool_option_notice"]["count"], 1)
            self.assertEqual(len(binding["targets"]), 3)


if __name__ == "__main__":
    unittest.main()
