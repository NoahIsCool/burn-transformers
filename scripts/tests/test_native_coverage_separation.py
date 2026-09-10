"""Opt-in, tiny native regressions for compiler coverage-separation boundaries.

Run with ``TFM_COVERAGE_SEPARATION_NATIVE_TESTS=1 python3 -m unittest discover
-s scripts/tests -p test_native_coverage_separation.py -v`` from the crate.
``TFM_COVERAGE_SEPARATION_PROOF_DIR`` optionally names a NEW directory in which
to retain commands, sources, binaries, profiles, native exports and identities.
Otherwise all native artifacts live in an automatically removed temporary tree.
Sources are embedded so the existing package rule includes the entire fixture.
No Cargo/Burn build, runtime dependency, coverage exclusion or release policy is
introduced. These observations do not qualify production-only measurements.

The exact producer is shared with coverage_binding.py, not inferred from the
mutable ``nightly`` alias. LLVM tools must come from that compiler's sysroot and
match its LLVM version. Each subprocess has a finite 30-second timeout.

Primary semantics: the nearest explicit coverage attribute wins, otherwise
lookup inherits the parent setting (default on at the crate root):
https://doc.rust-lang.org/stable/nightly-rustc/src/rustc_mir_transform/coverage/query.rs.html
https://doc.rust-lang.org/rustc/instrument-coverage.html
Those documentation URLs are rolling; the executable regressions pin the
actual compiler. Module support and downstream generic behavior are tested,
not inferred solely from the unstable book's function-oriented description.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1]
with mock.patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    from coverage_binding import producer_from_version
    from coverage_provenance import fingerprint_file


SOURCES = {
    "module_off.rs": """#![feature(coverage_attribute)]
pub fn runtime(value: u32) -> u32 { // runtime body
    if value > 0 { value + 1 } else { 0 }
}
#[coverage(off)]
#[cfg(test)]
mod tests {
    macro_rules! generated {
        () => {
            fn generated_helper() -> u32 { 2 }
            #[test]
            fn generated_test() { assert_eq!(super::runtime(1), generated_helper()); }
        };
    }
    generated!();
    fn helper() -> u32 { 3 }
    #[test]
    fn ordinary_test() {
        let closure = || helper();
        assert_eq!(super::runtime(2), closure());
    }
}
""",
    "module_bypass.rs": """#![feature(coverage_attribute)]
pub fn runtime(value: u32) -> u32 { value + 1 } // runtime body
#[coverage(off)]
#[cfg(test)]
mod tests {
    macro_rules! generated_override {
        () => {
            #[coverage(on)]
            fn reenabled_helper() -> u32 { 2 } // reenabled body
            #[test]
            fn hidden_test() { assert_eq!(super::runtime(1), reenabled_helper()); }
        };
    }
    generated_override!();
}
macro_rules! outside {
    () => {
        #[cfg(test)]
        fn outside_helper() -> u32 { 3 } // outside helper body
        #[cfg(test)]
        #[test]
        fn outside_test() { assert_eq!(runtime(2), outside_helper()); } // outside test body
    };
}
outside!();
""",
    "origin.rs": """#![allow(dead_code)]
macro_rules! generated {
    () => {
        #[cfg(not(test))]
        fn selected() -> u32 { 1 } // production origin
        #[cfg(test)]
        fn selected() -> u32 { 1 } // test origin
        #[cfg(test)]
        #[test]
        fn hidden_test() { assert_eq!(public(0), 0); } // generated test origin
    };
}
generated!();
pub fn public(value: u32) -> u32 {
    if value > 0 { value + selected() } else { 0 }
}
pub fn unused(value: u32) -> u32 { value * 17 } // unused runtime body
""",
    "external.rs": """#[test]
fn external_exercises_real_library() {
    assert_eq!(origin::public(2), 3);
}
""",
    "generic_origin.rs": """pub fn used(value: u32) -> u32 { value + 1 } // nongeneric body
pub fn unused_generic<T>(value: T) -> T { value } // unused generic body
pub fn used_generic<T>(value: T) -> T { value } // used generic body
""",
    "generic_external.rs": """#[test]
fn external_exercises_real_library() {
    assert_eq!(generic_origin::used(2), 3);
    assert_eq!(generic_origin::used_generic(5_u32), 5);
}
""",
}


def marker_line(source, marker):
    """Resolve a unique physical source line without hard-coding its number."""

    lines = [index for index, line in enumerate(SOURCES[source].splitlines(), 1) if marker in line]
    if len(lines) != 1:
        raise ValueError(f"fixture marker must be unique: {source}: {marker}")
    return lines[0]


def llvm_version(output):
    """Preserve the complete LLVM/Rust build version, not just its numeric prefix."""

    versions = re.findall(r"^\s*LLVM version (\d+\.\d+\.\d+(?:-rust-\d+\.\d+\.\d+-nightly)?)\s*$", output, re.MULTILINE)
    if len(versions) != 1:
        raise ValueError("missing or ambiguous LLVM tool version")
    return versions[0]


class SeparationFixtureTests(unittest.TestCase):
    """Keep fixture lookup/version checks runnable without a native toolchain."""

    def test_version_is_exact_and_unambiguous(self):
        """Truncation, duplicates, vendor labels and suffixes cannot prove a pin."""

        self.assertEqual(llvm_version("LLVM\n  LLVM version 21.1.8\n  Optimized build.\n"), "21.1.8")
        self.assertEqual(llvm_version("  LLVM version 21.1.8-rust-1.94.0-nightly\n"), "21.1.8-rust-1.94.0-nightly")
        for value in ("", "LLVM version 21.1", "LLVM version 21.1.8-extra",
                      "Apple LLVM version 21.1.8", "LLVM version 21.1.8\nLLVM version 21.1.8"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                llvm_version(value)

    def test_identical_bodies_do_not_identify_physical_source(self):
        """Equal expanded bodies can still originate from distinct cfg branches."""

        production = marker_line("origin.rs", "// production origin")
        testing = marker_line("origin.rs", "// test origin")
        lines = SOURCES["origin.rs"].splitlines()
        self.assertNotEqual(production, testing)
        self.assertEqual(lines[production - 1].partition("//")[0], lines[testing - 1].partition("//")[0])
        with self.assertRaises(ValueError):
            marker_line("origin.rs", "missing marker")
        with self.assertRaises(ValueError):
            marker_line("origin.rs", "fn selected")


@unittest.skipUnless(os.environ.get("TFM_COVERAGE_SEPARATION_NATIVE_TESTS") == "1",
                     "opt-in tiny pinned native Rust/LLVM separation fixture")
class NativeSeparationTests(unittest.TestCase):
    """Test native mappings, including counterexamples to overbroad exclusion."""

    @classmethod
    def setUpClass(cls):
        """Compile eight tiny artifacts once, retaining independent native exports."""

        proof = os.environ.get("TFM_COVERAGE_SEPARATION_PROOF_DIR")
        if proof:
            cls.root = Path(proof).resolve()
            cls.root.mkdir()  # Never replace an earlier proof or silently reuse outputs.
        else:
            temporary = tempfile.TemporaryDirectory(prefix="tfm native separation ")
            cls.addClassCleanup(temporary.cleanup)
            cls.root = Path(temporary.name).resolve()
        cls.commands = []
        cls.environment = {key: value for key, value in os.environ.items()
                           if not key.startswith(("TFM_", "RUST", "CARGO_", "LLVM_"))}
        cls.environment.update({"CARGO_INCREMENTAL": "0", "CARGO_TERM_COLOR": "never"})
        cls.rustc = Path(cls.native("rustc-path", ["rustup", "which", "--toolchain", "nightly", "rustc"]).strip()).resolve(strict=True)
        version = cls.native("rustc-version", [cls.rustc, "-Vv"])
        cls.producer = producer_from_version(version)
        fields = dict(line.split(": ", 1) for line in version.splitlines()[1:])
        sysroot = Path(cls.native("sysroot", [cls.rustc, "--print", "sysroot"]).strip()).resolve(strict=True)
        llvm = sysroot / "lib" / "rustlib" / fields["host"] / "bin"
        cls.cov, cls.profdata = (llvm / name for name in ("llvm-cov", "llvm-profdata"))
        for name, tool in (("cov", cls.cov), ("profdata", cls.profdata)):
            expected_llvm = cls.producer.llvm_version + "-rust-" + cls.producer.release
            if llvm_version(cls.native(name + "-version", [tool, "--version"])) != expected_llvm:
                raise ValueError("LLVM tool version does not match the pinned compiler")
        cls.tools = {str(tool): fingerprint_file(tool, maximum_bytes=512 * 1024 * 1024)
                     for tool in (cls.rustc, cls.cov, cls.profdata)}
        for name, source in SOURCES.items():
            (cls.root / name).write_text(source, encoding="utf-8")
        cls.reports = {}
        for name in ("module_off", "module_bypass"):
            binary = cls.compile(name, name + ".rs", "--test")
            profile = cls.execute(name, binary, 2)
            cls.reports[name] = cls.export(name, [binary], [profile])

        library = cls.compile("liborigin.rlib", "origin.rs", "--crate-type=rlib", "--crate-name=origin")
        unit = cls.compile("origin_unit", "origin.rs", "--test", "--crate-name=origin")
        external = cls.compile("external", "external.rs", "--test", "--extern", "origin=" + str(library))
        unit_profile = cls.execute("origin_unit", unit, 1)
        external_profile = cls.execute("external", external, 1)
        profiles = [unit_profile, external_profile]
        cls.reports["origin_all"] = cls.export("origin_all", [external, unit], profiles)
        cls.reports["origin_external"] = cls.export("origin_external", [external], profiles)

        library = cls.compile("libgeneric_origin.rlib", "generic_origin.rs", "--crate-type=rlib", "--crate-name=generic_origin")
        common = ["--test", "--extern", "generic_origin=" + str(library)]
        for name, instrument, attributes in (
            ("generic_injected", True, ["-Zcrate-attr=feature(coverage_attribute)", "-Zcrate-attr=coverage(off)"]),
            ("generic_uninstrumented", False, []),
        ):
            binary = cls.compile(name, "generic_external.rs", *common, *attributes, instrument=instrument)
            profile = cls.execute(name, binary, 1)
            cls.reports[name] = cls.export(name, [binary], [profile])
        cls.write_proof()

    @classmethod
    def native(cls, name, arguments, profile=None):
        """Require warning-free native success and retain exact commands/log bytes."""

        argv = [str(argument) for argument in arguments]
        environment = dict(cls.environment)
        if profile is not None:
            environment["LLVM_PROFILE_FILE"] = str(profile)
        stdout, stderr = (cls.root / (name + suffix) for suffix in (".stdout", ".stderr"))
        with stdout.open("xb") as out, stderr.open("xb") as err:
            result = subprocess.run(argv, cwd=cls.root, env=environment, stdout=out, stderr=err, timeout=30)
        cls.commands.append({"command": argv, "cwd": str(cls.root), "timeout_seconds": 30,
                             "process_exit_code": result.returncode, "profile": str(profile) if profile else None,
                             "stdout": {"path": str(stdout), **fingerprint_file(stdout)},
                             "stderr": {"path": str(stderr), **fingerprint_file(stderr)}})
        with (cls.root / "commands.json").open("w", encoding="utf-8") as stream:
            json.dump(cls.commands, stream, indent=2)
            stream.write("\n")
        if result.returncode != 0 or stderr.stat().st_size:
            raise AssertionError(f"native command failed or warned: {argv}\n{stdout.read_text()}\n{stderr.read_text()}")
        return stdout.read_text(encoding="utf-8")

    @classmethod
    def compile(cls, name, source, *arguments, instrument=True):
        """Compile direct rustc inputs with explicit instrumentation and codegen."""

        output = cls.root / name
        cls.native(name + "-build", [cls.rustc, "--edition=2024", "-Dwarnings", "-Copt-level=0", "-Ccodegen-units=1",
                                  "-Cinstrument-coverage=" + ("yes" if instrument else "no"),
                                  *arguments, cls.root / source, "-o", output])
        return output

    @classmethod
    def execute(cls, name, binary, tests):
        """Bind each profile to a fresh binary and require every expected test pass."""

        profile = cls.root / (name + ".profraw")
        output = cls.native(name + "-run", [binary, "--test-threads=1"], profile)
        if f"test result: ok. {tests} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;" not in output:
            raise AssertionError("unexpected native test identities/counts: " + output)
        if not profile.is_file() or not profile.stat().st_size:
            raise AssertionError("native execution did not emit a profile")
        return profile

    @classmethod
    def export(cls, name, objects, profiles):
        """Export the exact object/profile lists with no source or function filter."""

        merged = cls.root / (name + ".profdata")
        cls.native(name + "-merge", [cls.profdata, "merge", "-sparse", *profiles, "-o", merged])
        extra = [value for binary in objects[1:] for value in ("-object", binary)]
        return json.loads(cls.native(name + "-export", [cls.cov, "export", objects[0], *extra, "-instr-profile=" + str(merged)]))

    @classmethod
    def write_proof(cls):
        """Retain observed bytes, never a release qualification or a test verdict."""

        proof = {"schema_version": 1, "purpose": "tiny_native_regression_not_release_or_exclusion_evidence",
                 "production_only_exclusions_qualified": False, "producer": cls.producer._asdict(),
                 "test_module": {"path": str(Path(__file__).resolve()), **fingerprint_file(Path(__file__).resolve())},
                 "tools": cls.tools, "commands": cls.commands,
                 "artifacts": {path.name: fingerprint_file(path) for path in sorted(cls.root.iterdir()) if path.is_file()}}
        with (cls.root / "observations.json").open("x", encoding="utf-8") as stream:
            json.dump(proof, stream, indent=2)
            stream.write("\n")

    def functions(self, report):
        """Read function records from the exact unmodified native export."""

        return [function for block in self.reports[report]["data"] for function in block["functions"]]

    def counts(self, report, source, marker):
        """Identify counters by physical origin, not Rust instantiating-crate names."""

        line = marker_line(source, marker)
        path = str(self.root / source)
        return sorted(function["count"] for function in self.functions(report)
                      if any(region[0] == line and function["filenames"][region[5]] == path
                             for region in function["regions"]))

    def test_module_off_inherits_into_tests_helpers_closures_and_macros(self):
        """Both tests execute but the sole mapped function is production runtime."""

        self.assertEqual(len(self.functions("module_off")), 1)
        self.assertEqual(self.counts("module_off", "module_off.rs", "// runtime body"), [2])

    def test_nested_on_override_escapes_module_off(self):
        """An opaque macro can explicitly re-enable a helper under an off module."""

        self.assertEqual(self.counts("module_bypass", "module_bypass.rs", "// reenabled body"), [1])
        self.assertFalse(any("hidden_test" in function["name"] for function in self.functions("module_bypass")))

    def test_generated_test_outside_annotated_module_remains_instrumented(self):
        """Annotating known test modules is not a proof about every generated item."""

        self.assertEqual(self.counts("module_bypass", "module_bypass.rs", "// outside helper body"), [1])
        self.assertEqual(self.counts("module_bypass", "module_bypass.rs", "// outside test body"), [1])
        self.assertEqual(self.counts("module_bypass", "module_bypass.rs", "// runtime body"), [2])
        self.assertEqual(len(self.functions("module_bypass")), 4)

    def test_root_injected_off_keeps_imported_generic_and_unused_runtime(self):
        """Definition-site exclusion removes the driver, not library instantiations."""

        self.assertEqual(self.counts("generic_injected", "generic_origin.rs", "// nongeneric body"), [1])
        self.assertEqual(self.counts("generic_injected", "generic_origin.rs", "// used generic body"), [0, 1])
        self.assertEqual(self.counts("generic_injected", "generic_origin.rs", "// unused generic body"), [0])
        # Mangled imported generic symbols include the instantiating crate name;
        # their source filenames, not that name substring, establish the origin.
        self.assertTrue(all(function["filenames"] == [str(self.root / "generic_origin.rs")]
                            for function in self.functions("generic_injected")))

    def test_disabling_instrumentation_loses_executed_generic_counters(self):
        """The same successful driver calls the generic, but only its zero map stays."""

        self.assertEqual(self.counts("generic_uninstrumented", "generic_origin.rs", "// nongeneric body"), [1])
        self.assertEqual(self.counts("generic_uninstrumented", "generic_origin.rs", "// used generic body"), [0])
        self.assertEqual(self.counts("generic_uninstrumented", "generic_origin.rs", "// unused generic body"), [0])
        self.assertTrue(all(function["filenames"] == [str(self.root / "generic_origin.rs")]
                            for function in self.functions("generic_uninstrumented")))

    def test_equal_macro_bodies_have_distinct_native_source_origins(self):
        """Identical expanded bodies do not authorize moving counts between lines."""

        self.assertEqual(self.counts("origin_all", "origin.rs", "// production origin"), [1])
        self.assertEqual(self.counts("origin_all", "origin.rs", "// test origin"), [0])
        self.assertEqual(self.counts("origin_all", "origin.rs", "// generated test origin"), [1])
        self.assertEqual(self.counts("origin_external", "origin.rs", "// production origin"), [1])
        self.assertEqual(self.counts("origin_external", "origin.rs", "// test origin"), [])
        self.assertEqual(self.counts("origin_external", "origin.rs", "// generated test origin"), [])
        self.assertEqual(self.counts("origin_external", "origin.rs", "// unused runtime body"), [0])


if __name__ == "__main__":
    unittest.main()
