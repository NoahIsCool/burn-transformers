"""Strict dep-info grammar and opt-in exact stable/coverage-nightly differentials.

Run ordinary tests without a compiler using unittest discovery. Enable the
isolated native matrix with ``TFM_DEPINFO_NATIVE_TESTS=1``. It runs the same tiny
metadata/dep-info fixtures with ``rustc +stable`` and ``rustc +nightly``, requiring
each exact release/commit/LLVM identity plus zero warnings or other stderr.
No Cargo builds, GPUs, live profiles, coverage exclusions or network are used.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "coverage_dependencies.py"
SPEC = importlib.util.spec_from_file_location("coverage_dependencies", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DEPENDENCIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPENDENCIES)
NATIVE_PRODUCERS = (
    ("+stable", DEPENDENCIES.STABLE_PRODUCER),
    ("+nightly", DEPENDENCIES.COVERAGE_PRODUCER),
)


def document(paths=("src/lib.rs",), *, targets="out.d out.rmeta", environment=""):
    """Construct rustc's output and phony rules, escaping only literal spaces."""

    escaped = [path.replace(" ", "\\ ") for path in paths]
    return f"{targets}: {' '.join(escaped)}\n\n" + "".join(
        f"{path}:\n" for path in escaped
    ) + environment


class DependencyGrammarTests(unittest.TestCase):
    """Exercise supported syntax, ambiguity rejection and inventory consistency."""

    def test_only_exact_reviewed_producers_are_supported(self):
        """Neither matching semver, a commit prefix nor another LLVM version is enough."""

        self.assertEqual(len(DEPENDENCIES.SUPPORTED_PRODUCERS), 2)
        for producer in DEPENDENCIES.SUPPORTED_PRODUCERS:
            self.assertEqual(DEPENDENCIES.require_supported_producer(**producer._asdict()), producer)
            for field, value in (("release", "1.94.0-nightly-other"),
                                 ("commit_hash", producer.commit_hash[:10]),
                                 ("commit_hash", "0" * 40), ("llvm_version", "21.1.9")):
                with self.subTest(producer=producer.release, field=field, value=value):
                    identity = producer._asdict()
                    identity[field] = value
                    with self.assertRaises(DEPENDENCIES.DependencyInfoError):
                        DEPENDENCIES.require_supported_producer(**identity)

    def test_syntax_does_not_imply_compiler_identity(self):
        """Successfully parsing unversioned text makes no source or producer assertion."""

        result = DEPENDENCIES.parse_dep_info(document())
        self.assertEqual(result.dependencies, ("src/lib.rs",))
        self.assertNotIn("producer", result._fields)

    def test_review_provenance_matches_the_explicit_producer_matrix(self):
        """Recorded source review pins remain aligned without pretending to prove exclusions."""

        fixture = Path(__file__).parent / "fixtures/coverage_dependencies/provenance.json"
        evidence = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertEqual(evidence["schema_version"], 1)
        self.assertIs(evidence["qualified_for_coverage_exclusions"], False)
        identities = tuple(DEPENDENCIES.RustcProducer(
            producer["release"], producer["commit_hash"], producer["llvm_version"]
        ) for producer in evidence["producers"])
        self.assertEqual(identities, DEPENDENCIES.SUPPORTED_PRODUCERS)
        for producer in evidence["producers"]:
            self.assertIn(f'/{producer["commit_hash"]}/', producer["source_url"])
        self.assertEqual(len(evidence["reviewed_byte_identical_sections"]), 3)

    def test_multiple_targets_and_outputs_are_not_dependencies(self):
        """Spaced/raw output targets do not contaminate compiler input paths."""

        text = document(("src/lib.rs", "fixture data.json"))
        text = text.replace("\n\n", "\nsecond output.rmeta: src/lib.rs fixture\\ data.json\n\n", 1)
        result = DEPENDENCIES.parse_dep_info(text)
        self.assertEqual(result.dependencies, ("src/lib.rs", "fixture data.json"))
        self.assertEqual(result.output_rule_count, 2)
        self.assertEqual(result.environment, ())

    def test_escaped_spaces_and_unicode_are_preserved(self):
        """Leading/trailing/repeated spaces and Unicode never undergo split or normalization."""

        paths = ("/Volumes/bag of holding/src/a.rs", " leading ", "two  spaces.rs", "雪/e\u0301.rs")
        result = DEPENDENCIES.parse_dep_info(document(paths))
        self.assertEqual(result.dependencies, paths)

    def test_duplicate_dependencies_are_returned_once(self):
        """Repeated tracked inputs and phony rules preserve first-seen identity."""

        result = DEPENDENCIES.parse_dep_info(document(("b.rs", "a.rs", "b.rs")))
        self.assertEqual(result.dependencies, ("b.rs", "a.rs"))

    def test_between_token_continuations_and_crlf(self):
        """Only explicitly separated, indented rule wrapping is supported."""

        text = "out.d out.rmeta: a.rs \\\n  with\\ space.rs \\\n    b.rs\n\na.rs:\nwith\\ space.rs:\nb.rs:\n"
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=repr(newline)):
                result = DEPENDENCIES.parse_dep_info(text.replace("\n", newline))
                self.assertEqual(result.dependencies, ("a.rs", "with space.rs", "b.rs"))

    def test_final_newline_is_optional(self):
        """A complete terminal phony rule needs no trailing physical newline."""

        self.assertEqual(DEPENDENCIES.parse_dep_info(document().rstrip("\n")).dependencies,
                         ("src/lib.rs",))

    def test_environment_uses_separate_escape_rules(self):
        """Spaces, Make punctuation, equals, CR/LF and backslashes remain environment data."""

        environment = (
            "\n# env-dep:ABSENT\n# env-dep:EMPTY=\n"
            "# env-dep:VALUE=a b#c$d=e\\\\f\\nnext\\rlast\n"
            "# env-dep:KEY\\nLINE=tab\tvalue\n"
        )
        result = DEPENDENCIES.parse_dep_info(document(environment=environment))
        self.assertEqual(result.environment, (
            DEPENDENCIES.EnvironmentDependency("ABSENT", None),
            DEPENDENCIES.EnvironmentDependency("EMPTY", ""),
            DEPENDENCIES.EnvironmentDependency("VALUE", "a b#c$d=e\\f\nnext\rlast"),
            DEPENDENCIES.EnvironmentDependency("KEY\nLINE", "tab\tvalue"),
        ))
        self.assertEqual(result.dependencies, ("src/lib.rs",))

    def test_environment_backslash_does_not_continue_comment(self):
        """A doubled terminal slash belongs to the environment, not the next line."""

        result = DEPENDENCIES.parse_dep_info(document(environment=
            "# env-dep:ONE=slash\\\\\n# env-dep:TWO=next\n"))
        self.assertEqual(result.environment, (("ONE", "slash\\"), ("TWO", "next")))

    def test_identical_environment_duplicates_do_not_conflict(self):
        """Exact duplicate observations are harmless, unlike changed values."""

        result = DEPENDENCIES.parse_dep_info(document(environment=
            "# env-dep:KEY=value\n# env-dep:KEY=value\n"))
        self.assertEqual(result.environment, (("KEY", "value"),))

    def test_native_special_filenames_and_generic_make_escapes_are_rejected(self):
        """Never reinterpret rustc's raw #/$/backslash as conventional Make escaping."""

        for path in ("hash#file.rs", "money$file.rs", r"slash\file.rs", r"hash\#file.rs",
                     "money$$file.rs", r"slash\\file.rs", "tab\tfile.rs", "C:/file.rs"):
            with self.subTest(path=path):
                with self.assertRaises(DEPENDENCIES.DependencyInfoError):
                    DEPENDENCIES.parse_dep_info(document((path,)))

    def test_make_constructs_fail_closed(self):
        """Variables, comments, recipes, assignments, patterns and order-only rules fail."""

        for rule in ("out: $(INPUT)", "out: a.rs # trailing", "out: a.rs; echo secret",
                     "out: a.rs | stamp", "out:: a.rs", "out: *.rs", "out: a?.rs",
                     "out: [ab].rs", "%.o: a.rs", "out: a.rs=bad", "out &: a.rs",
                     "archive(member): a.rs", "out: ~/a.rs", "\techo recipe"):
            with self.subTest(rule=rule):
                with self.assertRaises(DEPENDENCIES.DependencyInfoError):
                    DEPENDENCIES.parse_dep_info(rule + "\na.rs:\n")

    def test_continuation_ambiguities_fail_closed(self):
        """Mid-token splits, missing indentation, comments and dangling slashes fail."""

        for text in ("out: a\\\n b.rs\n", "out: a.rs \\", "out: a.rs \\\n",
                     "out: a.rs \\\nb.rs\n", "out: a.rs \\\n # env-dep:KEY=value\n",
                     "out: a.rs \\\n   \n", "out: a.rs\\\\\n b.rs\n"):
            with self.subTest(text=text):
                with self.assertRaises(DEPENDENCIES.DependencyInfoError):
                    DEPENDENCIES.parse_dep_info(text)

    def test_literal_terminal_backslash_cannot_swallow_an_input(self):
        """The corresponding phony rule vetoes a misleading escaped-space parse."""

        text = "out: slash\\ next.rs\nslash\\:\nnext.rs:\n"
        with self.assertRaises(DEPENDENCIES.DependencyInfoError):
            DEPENDENCIES.parse_dep_info(text)

    def test_escaped_space_is_not_a_continuation_separator(self):
        """A quoted terminal space cannot merge the next physical line into a path."""

        text = "out: a\\ \\\n b.rs\na\\ b.rs:\n"
        with self.assertRaises(DEPENDENCIES.DependencyInfoError):
            DEPENDENCIES.parse_dep_info(text)

    def test_unexplained_comments_fail_closed(self):
        """Unsupported comments cannot hide filenames or unparsed checksum metadata."""

        for comment in ("# ordinary comment", "# checksum:sha256=123 file_len:1 a.rs",
                        " # env-dep:KEY=value", "#env-dep:KEY=value"):
            with self.subTest(comment=comment):
                with self.assertRaises(DEPENDENCIES.DependencyInfoError):
                    DEPENDENCIES.parse_dep_info(document() + comment + "\n")

    def test_environment_values_are_not_leaked_by_errors(self):
        """Malformed escapes/conflicts provide context without reflecting secret values."""

        for suffix in ("# env-dep:KEY=SECRET\\q\n", "# env-dep:KEY=SECRET\\\n",
                       "# env-dep:=SECRET\n", "# env-dep:KEY=SECRET\n# env-dep:KEY=other\n",
                       "# env-dep:KEY\n# env-dep:KEY=SECRET\n"):
            with self.subTest(suffix=suffix):
                with self.assertRaises(DEPENDENCIES.DependencyInfoError) as caught:
                    DEPENDENCIES.parse_dep_info(document() + suffix, context="fixture.d")
                self.assertIn("fixture.d: line", str(caught.exception))
                self.assertNotIn("SECRET", str(caught.exception))

    def test_output_and_phony_inventories_must_agree(self):
        """Truncation, hidden extra inputs, and cross-output disagreement are rejected."""

        for text in ("out: a.rs\n", "out: a.rs\nb.rs:\n", "out: a.rs\na.rs:\nb.rs:\n",
                     "one: a.rs\ntwo: b.rs\na.rs:\nb.rs:\n",
                     "one: a.rs b.rs\ntwo: b.rs a.rs\na.rs:\nb.rs:\n",
                     "out: a.rs b.rs\na.rs b.rs:\n"):
            with self.subTest(text=text):
                with self.assertRaises(DEPENDENCIES.DependencyInfoError):
                    DEPENDENCIES.parse_dep_info(text)

    def test_missing_targets_rules_and_phase_inversions_fail(self):
        """Empty or reordered fragments are not accepted as a complete rustc inventory."""

        for text in ("", " \n\n", "a.rs:\n", ": a.rs\na.rs:\n", "not a rule\n",
                     "a.rs:\nout: a.rs\n", "out: a.rs\na.rs:\nother: a.rs\n",
                     "# env-dep:KEY=value\n" + document(),
                     document() + "# env-dep:KEY=value\nother: a.rs\n"):
            with self.subTest(text=text):
                with self.assertRaises(DEPENDENCIES.DependencyInfoError):
                    DEPENDENCIES.parse_dep_info(text)

    def test_invalid_input_encoding_and_control_characters_fail(self):
        """Lossy decoding and forbidden physical line encodings cannot change identity."""

        for text in (None, b"out: a.rs", document(("nul\x00.rs",)),
                     document(("lossy\ufffd.rs",)), document(("bom\ufeff.rs",)),
                     document(("surrogate\udcff.rs",)), document(("del\x7f.rs",)),
                     document().replace("\n", "\r")):
            with self.subTest(text=repr(text)):
                with self.assertRaises(DEPENDENCIES.DependencyInfoError):
                    DEPENDENCIES.parse_dep_info(text)

    def test_paths_remain_opaque_not_resolved_or_classified(self):
        """Absolute/traversing/test-named inputs are returned, not trusted or excluded."""

        paths = ("/outside/project.rs", "../shared.rs", "src/cfg_test.rs")
        self.assertEqual(DEPENDENCIES.parse_dep_info(document(paths)).dependencies, paths)


@unittest.skipUnless(os.environ.get("TFM_DEPINFO_NATIVE_TESTS") == "1",
                     "set TFM_DEPINFO_NATIVE_TESTS=1 for the pinned native compiler matrix")
class NativeRustcDependencyTests(unittest.TestCase):
    """Compare identical fixtures under the exact stable and coverage producers."""

    @classmethod
    def setUpClass(cls):
        """Require the exact installed producer instead of silently testing another version."""

        for toolchain, producer in NATIVE_PRODUCERS:
            result = subprocess.run(["rustc", toolchain, "-Vv"], capture_output=True,
                                    text=True, check=False, timeout=20)
            if result.returncode != 0 or result.stderr:
                raise AssertionError(f"{toolchain}: version command failed or emitted diagnostics")
            for field, value in (("release", producer.release),
                                 ("commit-hash", producer.commit_hash),
                                 ("LLVM version", producer.llvm_version)):
                if result.stdout.splitlines().count(f"{field}: {value}") != 1:
                    raise AssertionError(f"{toolchain}: compiler does not match the pinned {field}")
            DEPENDENCIES.require_supported_producer(**producer._asdict())

    def compile_fixture(self, toolchain, root, files, environment=None):
        """Emit only tiny metadata/dep-info in an isolated temporary directory."""

        for name, contents in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        outputs = root / "compiler outputs"
        outputs.mkdir()
        env = dict(os.environ)
        env.pop("TFM_DEPINFO_NATIVE_ABSENT_79137", None)
        env.update(environment or {})
        result = subprocess.run([
            "rustc", toolchain, "--edition=2024", "--crate-type=lib", "--crate-name=dep_fixture",
            "entry point.rs", "--emit=metadata=compiler outputs/fixture.rmeta,"
            "dep-info=compiler outputs/fixture.d",
        ], cwd=root, env=env, capture_output=True, text=True, check=False, timeout=30)
        self.assertEqual(result.returncode, 0, "native fixture compiler failed")
        self.assertEqual(result.stderr, "", "native fixture emitted compiler diagnostics")
        self.assertEqual(result.stdout, "")
        return (outputs / "fixture.d").read_text(encoding="utf-8")

    def test_native_spaces_in_sources_includes_outputs_and_environment(self):
        """Actual raw spaced targets and escaped inputs preserve all source identities."""

        files = {
            "entry point.rs": '#[path = "nested module.rs"]\nmod nested;\n'
                'pub const VALUE: usize = nested::VALUE;\n'
                'pub const DATA: &str = include_str!("fixture data.txt");\n'
                'pub const ABSENT: Option<&str> = option_env!("TFM_DEPINFO_NATIVE_ABSENT_79137");\n'
                'pub const PRESENT: &str = env!("TFM_DEPINFO_NATIVE_PRESENT_79137");\n',
            "nested module.rs": "pub const VALUE: usize = 7;\n",
            "fixture data.txt": "test fixture\n",
        }
        value = "a b#c$d=e\\f\nnext\rlast"
        emissions = []
        for toolchain, _ in NATIVE_PRODUCERS:
            with self.subTest(toolchain=toolchain), tempfile.TemporaryDirectory(prefix="tfm dep native ") as directory:
                emitted = self.compile_fixture(toolchain, Path(directory), files,
                                               {"TFM_DEPINFO_NATIVE_PRESENT_79137": value})
                self.assertIn("compiler outputs/fixture.d: entry\\ point.rs", emitted)
                result = DEPENDENCIES.parse_dep_info(emitted)
                self.assertEqual(set(result.dependencies), set(files))
                self.assertEqual(result.output_rule_count, 2)
                self.assertEqual(dict(result.environment), {
                    "TFM_DEPINFO_NATIVE_ABSENT_79137": None,
                    "TFM_DEPINFO_NATIVE_PRESENT_79137": value,
                })
                emissions.append(emitted)
        self.assertEqual(emissions[0], emissions[1], "pinned producer outputs differ")

    def test_native_unescaped_special_paths_are_rejected_not_misdecoded(self):
        """The real producer writes raw #/$/backslash paths; do not invent Make escapes."""

        for name in ("hash#file.txt", "money$file.txt", "slash\\file.txt"):
            files = {
                "entry point.rs": f'pub const DATA: &str = include_str!(r"{name}");\n',
                name: "test fixture\n",
            }
            emissions = []
            for toolchain, _ in NATIVE_PRODUCERS:
                with self.subTest(toolchain=toolchain, name=name), tempfile.TemporaryDirectory(prefix="tfm-dep-special-") as directory:
                    emitted = self.compile_fixture(toolchain, Path(directory), files)
                    self.assertIn(f" {name}\n", emitted)
                    self.assertIn(f"\n{name}:\n", emitted)
                    with self.assertRaises(DEPENDENCIES.DependencyInfoError):
                        DEPENDENCIES.parse_dep_info(emitted)
                    emissions.append(emitted)
            self.assertEqual(emissions[0], emissions[1], "pinned producer outputs differ")


if __name__ == "__main__":
    unittest.main()
