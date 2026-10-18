"""Validate the October 18 Nemotron-H envelope decision record without checkpoints, bundles, or a GPU.

The record is published with the crate. Its bundle-only run and
source-validation logs live in the repository-only campaign directory
``acceptance/measurements/nemotron-h-envelope-decision-20261018/``, and it
cites the envelope round's repository-only evidence, so every read of those
goes through :func:`repository_evidence.repository_evidence`; in the extracted
package those tests skip, and in a Git checkout a missing file fails them.
The tests tie the decision, the restored rule, the inferred verdict, the
diagnostic, every correction, and every check to the committed evidence;
require the gate source at the record's source commit (read from Git, so later
changes cannot invalidate the record) to judge by the two-graph floors without
compiled slacks; require that commit's library code to equal the envelope
round's gate source; and require the envelope record and the earlier records
to keep their bytes.
"""

import functools
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import unittest

from repository_evidence import PACKAGE_CONTENTS, repository_evidence


ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = ROOT / "acceptance/measurements"
RECORD = MEASUREMENTS / "nemotron-h-envelope-decision-20261018.json"
ENVELOPE_RECORD = MEASUREMENTS / "nemotron-h-envelope-20261018.json"
HALF_PRECISION_ROUND = MEASUREMENTS / "half-precision-envelope-20261018.json"
PARITY = "tests/support/nemotron_h_parity.rs"
ARTIFACTS = "tests/nemotron_h_artifacts.rs"
GENERATOR = ROOT / "scripts/generate_nemotron_h_f16_graph_reference.py"
#: Records the decision must leave untouched, the envelope record among them.
HISTORICAL = frozenset({
    "half-precision-calibration-20261017.json", "mac-validation-20261017.json",
    "half-precision-envelope-20261018.json", "half-precision-followups-20261017.json",
    "half-precision-review-20261018.json", "reference-precision-followups-20261016.json",
    "nemotron-h-envelope-20261018.json",
    "../nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json",
})
#: Row paths in gate order.
PATHS = ("prompt", "teacher", "cached")
#: The two official F16 evaluations whose worst rows are the gate's floors.
FLOOR_GRAPHS = ("mps-eager", "cpu-eager")
#: Source checks the standard validation requires, by record name.
REQUIRED_CHECKS = frozenset({
    "fmt", "clippy-default-all-features", "clippy-1.99.1-all-features", "clippy-1.99.1-backends",
    "test-default", "test-std-flex-train", "msrv-1.92.0-check", "rustdoc-private", "python-tests",
    "documentation-audit",
})


def sha256(data):
    """Hexadecimal SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def quoted_entries(value):
    """Every ``{"log": ..., "quotes": [...]}`` mapping anywhere in ``value``."""
    if isinstance(value, dict):
        if isinstance(value.get("log"), str) and isinstance(value.get("quotes"), list):
            yield value
        for child in value.values():
            yield from quoted_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from quoted_entries(child)


def git(*arguments):
    """Output of a Git command in the repository; skips where no Git checkout exists (the published crate)."""
    if PACKAGE_CONTENTS.is_packaged_archive(ROOT):
        raise unittest.SkipTest("this check needs the Git history of the repository")
    result = subprocess.run(["git", "-C", str(ROOT), *arguments], capture_output=True, text=True, check=False)
    if result.returncode:
        raise AssertionError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout


class NemotronEnvelopeDecisionRecordTests(unittest.TestCase):
    """Tie the decision record's statuses, figures, quotes, and identities to the committed evidence."""

    @classmethod
    def setUpClass(cls):
        """Read only the published records; campaign files are read on demand."""
        cls.record = json.loads(RECORD.read_text())
        cls.envelope = json.loads(ENVELOPE_RECORD.read_text())
        cls.campaign = cls.record["campaign_directory"]

    def measured(self, relative):
        """Bytes of a campaign file of this record or of the envelope record, checked against that record's ``files``."""
        campaign, _, inner = relative.partition("/")
        owner = {self.campaign: self.record, self.envelope["campaign_directory"]: self.envelope}[campaign]
        data = repository_evidence(MEASUREMENTS / relative).read_bytes()
        self.assertEqual(sha256(data), owner["files"][inner], relative)
        return data

    @functools.lru_cache(maxsize=None)
    def text(self, relative):
        """Text of an authenticated campaign file, relative to acceptance/measurements."""
        return self.measured(relative).decode()

    def own(self, relative):
        """Text of an authenticated file of this record's campaign directory."""
        return self.text(f"{self.campaign}/{relative}")

    def derivation(self):
        """The envelope round's authenticated derivation."""
        return json.loads(self.text(f"{self.envelope['campaign_directory']}/derivation/envelope.json"))

    def test_record_states_a_failing_open_finding_not_a_support_declaration(self):
        """The record promotes nothing, does not pass, and calls its verdict inferred."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertFalse(record["advertised_supported"])
        self.assertTrue(record["scope"].endswith("not_a_support_declaration"))
        self.assertFalse(record["passed"])
        self.assertEqual(record["verdict"]["status"], "fails: an open finding")
        self.assertEqual(record["verdict"]["basis"], "inferred")
        self.assertIn("not run on the checkpoint", record["verdict"]["basis_note"])
        self.assertTrue(any("30B checkpoint" in item for item in record["not_run"]))
        self.assertEqual(record["diagnostic"]["status"], "printed, not adopted")
        for key in ("base_commit", "gate_commit", "code_head", "documentation_commit", "source_commit"):
            self.assertRegex(record[key], r"^[0-9a-f]{40}$")
        self.assertEqual(record["extends"]["record"], ENVELOPE_RECORD.name)
        self.assertEqual(record["extends"]["sha256"], sha256(ENVELOPE_RECORD.read_bytes()))

    def test_decision_is_quoted_with_its_date(self):
        """The decision quote names the open finding, the rejected rule, and both reasons, dated October 18."""
        decision = self.record["decision"]
        self.assertEqual(decision["date"], "2026-10-18")
        quote = decision["quote"]
        for phrase in ("stays an open finding", "Do NOT adopt the four-graph envelope rule", "slack is 1.1",
                       "at most 1.0046", "cached decision 22 sits at 1.023 times the envelope"):
            self.assertIn(phrase, quote)
        largest = max(path["largest_ratio"] for path in self.derivation()["paths"].values())
        self.assertEqual(f"{largest:.4f}", "1.0046")
        self.assertEqual(f"{self.envelope['burn']['paths']['cached']['of_envelope']:.3f}", "1.023")

    def test_files_cover_the_campaign_directory(self):
        """``files`` lists exactly the campaign directory's files, each with its digest."""
        root = repository_evidence(MEASUREMENTS / self.campaign / "run_check.py").parent
        present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        self.assertEqual(present, set(self.record["files"]))
        for relative in sorted(present):
            with self.subTest(file=relative):
                self.measured(f"{self.campaign}/{relative}")

    def test_every_quote_is_in_its_log(self):
        """Every quote anywhere in the record is a line of the log it cites, or part of one."""
        for entry in quoted_entries(self.record):
            lines = self.text(entry["log"]).splitlines()
            for quote in entry["quotes"]:
                with self.subTest(log=entry["log"], quote=quote[:60]):
                    self.assertGreaterEqual(len(quote), 20)
                    self.assertTrue(any(quote in line for line in lines), quote)

    def test_floors_are_the_two_official_evaluations_worst_rows(self):
        """The measured floors are, per path, the worse minimum cosine of the oracle and the CPU/F16 rows."""
        derivation = self.derivation()
        log = self.own(self.record["runs"][0]["log"])
        printed = re.search(r"Nemotron official F16 floors \(the gate's rule\) \[prompt, teacher, cached\]: "
                            r"\[([0-9.]+), ([0-9.]+), ([0-9.]+)\]", log).groups()
        rule = self.record["rule"]
        for kind, value in zip(PATHS, printed):
            with self.subTest(path=kind):
                graphs = derivation["paths"][kind]["graphs"]
                worst = min(FLOOR_GRAPHS, key=lambda name: graphs[name]["minimum_cosine"])
                self.assertEqual(f"{graphs[worst]['minimum_cosine']:.9f}", value)
                self.assertEqual(rule["floors"][kind], float(value))
                self.assertEqual(rule["floor_rows"][kind]["graph"], worst)
                self.assertEqual(rule["floor_rows"][kind]["decision"], graphs[worst]["decision"])
                self.assertEqual(rule["floor_rows"][kind]["cosine"], graphs[worst]["minimum_cosine"])
                self.assertGreaterEqual(rule["floors"][kind], 0.99)
        self.assertEqual(self.envelope["derivation"]["previous_floor"], rule["floors"])

    def test_restored_rule_fails_only_at_cached_decision_22(self):
        """Burn's rows measured at 2cf83d7, judged by the floors, violate them at cached decision 22 alone."""
        burn = self.envelope["burn"]["paths"]
        floors = self.record["rule"]["floors"]
        verdict = self.record["verdict"]
        gate_log = self.text(f"{self.envelope['campaign_directory']}/gates/logs/"
                             "nemotron-h.f16.metal.official-precision.log")
        violations = []
        for kind in PATHS:
            with self.subTest(path=kind):
                cosines = burn[kind]["cosines"]
                below = [decision for decision, cosine in enumerate(cosines) if cosine < floors[kind]]
                violations += [(kind, decision) for decision in below]
                path = verdict["paths"][kind]
                self.assertEqual(path["rows_below_floor"], below)
                self.assertEqual(path["rows"], len(cosines))
                self.assertEqual((path["minimum_cosine"], path["decision"]),
                                 (burn[kind]["minimum_cosine"], burn[kind]["decision"]))
                self.assertAlmostEqual(path["margin"], burn[kind]["minimum_cosine"] - floors[kind], places=12)
                self.assertIn(f"{kind.capitalize()}: {len(cosines)} rows, minimum cosine "
                              f"{burn[kind]['minimum_cosine']:.12f} at decision {burn[kind]['decision']}", gate_log)
        self.assertEqual(violations, [("cached", 22)])
        (failing,) = verdict["failing_rows"]
        self.assertEqual((failing["path"], failing["decision"]), ("cached", 22))
        self.assertEqual(f"{failing['cosine']:.6f}", "0.998558")
        self.assertEqual(f"{failing['floor']:.6f}", "0.998635")
        self.assertLess(failing["cosine"], failing["floor"])
        earlier = next(entry for entry in json.loads(HALF_PRECISION_ROUND.read_text())["findings"]
                       if entry["id"] == "nemotron-floor-failure")
        measured = re.search(r"cosine ([0-9.]+) against the F32 truth is below the official F16 floor ([0-9.]+)",
                             earlier["evidence"][0]["quotes"][0]).groups()
        self.assertEqual(f"{float(measured[0]):.12f}", f"{failing['cosine']:.12f}")
        self.assertEqual(f"{float(measured[1]):.9f}", f"{failing['floor']:.9f}")

    def test_library_code_is_the_measured_gate_source(self):
        """The validated source compiles the library code of the 2cf83d7 gate run, so its rows apply."""
        measured = self.envelope["source_commit"]
        for commit in (measured, self.record["source_commit"], self.record["gate_commit"]):
            git("cat-file", "-e", f"{commit}^{{commit}}")
        changed = git("diff", "--name-only", measured, self.record["source_commit"], "--", "src", "Cargo.toml",
                      "Cargo.lock")
        self.assertEqual(changed, "")
        self.assertEqual(git("diff", "--name-only", self.record["gate_commit"], self.record["source_commit"], "--",
                             "src", "tests", "Cargo.toml", "Cargo.lock"), "")

    def test_gate_source_judges_by_the_floors_without_slacks(self):
        """At the source commit the gate applies the two-graph floors; the envelope is a slack-free diagnostic."""
        parity = git("show", f"{self.record['source_commit']}:{PARITY}")
        artifacts = git("show", f"{self.record['source_commit']}:{ARTIFACTS}")
        for name in ("PROMPT_SLACK", "TEACHER_SLACK", "CACHED_SLACK", "assert_leave_one_out", "assert_within_envelopes"):
            self.assertNotIn(name, parity)
            self.assertNotIn(name, artifacts)
        self.assertIn("pub(super) fn half_precision_floor(", parity)
        self.assertIn("pub(super) fn assert_within_floors(", parity)
        self.assertIn("pub(super) struct BurnEnvelopeRatio", parity)
        floors = artifacts[artifacts.index("fn official_f16_floors("):]
        floors = "".join(floors[:floors.index("\n}\n")].split())
        self.assertIn("half_precision_floor(", floors)
        self.assertIn("&[oracle.as_slice(),precision.matrix(Precision::Float16,kind),]", floors)
        self.assertIn('assert_within_floors("Nemotron Burn F16 against the F32 truth", &summaries);', artifacts)
        self.assertIn(f"fn {self.record['runs'][0]['test']}()", artifacts)

    def test_diagnostic_matches_the_derivation_and_burns_ratio(self):
        """The bundle-only run reprints the derivation's envelope; Burn's cached row exceeds every official ratio."""
        derivation = self.derivation()
        log = self.own(self.record["runs"][0]["log"])
        for kind in PATHS:
            with self.subTest(path=kind):
                path = derivation["paths"][kind]
                diagnostic = self.record["diagnostic"]["paths"][kind]
                self.assertEqual(diagnostic["envelope"], path["envelope"])
                self.assertEqual(diagnostic["envelope_graph"], path["envelope_graph"])
                self.assertEqual(diagnostic["largest_leave_one_out"], path["largest_ratio"])
                self.assertEqual(diagnostic["registered_method_slack_not_adopted"], 1.1)
                line = re.search(rf"\(not a gate\) {kind.capitalize()}: envelope ([0-9.e+-]+) \(([a-z-]+)\); "
                                 r"largest leave-one-out ([0-9.]+) \(([a-z-]+)\); the registered method's slack "
                                 r"([0-9.]+) \(cosine floor ([0-9.]+)\) is not adopted", log).groups()
                self.assertEqual(float(line[0]), float(f"{path['envelope']:.6e}"))
                self.assertEqual(line[1], path["envelope_graph"])
                self.assertEqual(line[2], f"{path['largest_ratio']:.4f}")
                self.assertEqual(line[3], path["largest_ratio_graph"])
                self.assertEqual(float(line[4]), path["slack"])
                self.assertEqual(line[5], f"{path['cosine_floor']:.9f}")
                for graph, worst in path["graphs"].items():
                    self.assertIn(f"{kind.capitalize()}: official {graph} worst delta", log)
                    self.assertIn(f"(cosine {worst['minimum_cosine']:.9f}) at decision {worst['decision']}", log)
                burn = self.envelope["burn"]["paths"][kind]
                self.assertEqual(diagnostic["burn_worst_of_envelope"], burn["of_envelope"])
                delta = math.sqrt(2.0 * (1.0 - burn["minimum_cosine"]))
                self.assertAlmostEqual(delta / path["envelope"], burn["of_envelope"], places=4)
                self.assertEqual(diagnostic["burn_beyond_every_official_graph"],
                                 burn["of_envelope"] > path["largest_ratio"])
        self.assertEqual([kind for kind in PATHS if self.record["diagnostic"]["paths"][kind]
                          ["burn_beyond_every_official_graph"]], ["cached"])

    def test_corrections_recompute_from_the_official_evidence(self):
        """The decision-30 cosines, the decision-18 spike, the hidden minimum, and the MPS budget are as corrected."""
        corrections = {entry["id"]: entry for entry in self.record["corrections"]}
        envelope_findings = {entry["id"]: entry for entry in self.envelope["findings"]}
        derivation = self.derivation()
        teacher = derivation["paths"]["teacher"]["graphs"]
        control = json.loads(self.text(f"{self.envelope['campaign_directory']}/bundles/"
                                       "f16-mps-eager-control/reference.json"))

        row30 = corrections["teacher-decision-30-official-cosines"]
        self.assertIn(row30["statement"], envelope_findings["teacher-decision-30"]["summary"])
        self.assertIn(row30["statement"], self.envelope["awaiting_maintainer"][2])
        expected = {name: graph["cosines"][30] for name, graph in teacher.items()}
        expected["mps-eager-control"] = control["comparisons"]["float32"]["teacher"]["cosines"][30]
        self.assertEqual(row30["teacher_decision_30_cosines"], expected)
        printed = {"cpu-eager": "0.999958", "cpu-sdpa": "0.999346", "mps-eager": "0.999968",
                   "mps-sdpa": "0.999954", "mps-eager-control": "0.999957"}
        self.assertEqual({name: f"{value:.6f}" for name, value in expected.items()}, printed)
        self.assertFalse(all(abs(value - 0.99996) < 5e-6 for value in expected.values()),
                         "the corrected statement was false")
        deficits = {name: 1.0 - value for name, value in expected.items() if name != "mps-eager-control"}
        others = [value for name, value in deficits.items() if name != "cpu-sdpa"]
        self.assertEqual((round(deficits["cpu-sdpa"] / max(others)), round(deficits["cpu-sdpa"] / min(others))),
                         (14, 21))
        burn = 1.0 - row30["burn_teacher_decision_30_cosine"]
        self.assertEqual(self.envelope["burn"]["paths"]["teacher"]["decision"], 30)
        self.assertEqual(round(burn / deficits["cpu-sdpa"]), 6)
        self.assertEqual((round(burn / max(others)), round(burn / min(others))), (85, 125))

        spike = corrections["teacher-spike-is-not-attention-path-alone"]
        self.assertIn("follows the attention path, not the device",
                      envelope_findings["teacher-spike-is-attention-path"]["summary"])
        row18 = {name: graph["cosines"][18] for name, graph in teacher.items()}
        self.assertEqual(spike["teacher_decision_18_cosines"], row18)
        self.assertEqual({name for name, value in row18.items() if value > 0.999}, {"cpu-eager"})
        self.assertTrue(all(value < 0.995 for name, value in row18.items() if name != "cpu-eager"))

        hidden = corrections["hidden-trace-minimum"]
        self.assertIn(hidden["statement"], envelope_findings["closer-to-the-f16-oracle"]["summary"])
        minimum = min(hidden["hidden_minimum_cosines"].values())
        self.assertEqual(minimum, hidden["hidden_minimum_cosines"]["cached_decode_hidden"])
        self.assertLess(minimum, 0.999995)
        self.assertGreaterEqual(minimum, 0.99999495)
        self.assertEqual(f"{minimum:.6f}", "0.999995")

        budget = corrections["mps-sdpa-resident-budget"]
        manifest = json.loads(self.text(f"{self.envelope['campaign_directory']}/bundles/f16-mps-sdpa/reference.json"))
        self.assertEqual(manifest["provenance"]["max_resident_gib"], 16.0)
        self.assertEqual(manifest["provenance"]["offload_execution_devices"], ["mps"])
        self.assertTrue(budget["evidence"])

        superseded = corrections["gate-pass-superseded"]
        self.assertTrue(self.envelope["passed"])
        self.assertTrue(superseded["correction"].startswith("The rule was reviewed and not adopted"))

        generator = corrections["generator-docstring-not-corrected"]
        source = GENERATOR.read_bytes()
        self.assertIn(generator["statement"], " ".join(source.decode().split()))
        self.assertEqual(sha256(source), next(graph["generator_sha256"] for graph in self.envelope["graphs"]
                                              if "generator_sha256" in graph))

    def test_bundle_only_run_is_clean_and_passed(self):
        """The bundle-only run ran once from the clean worktree at the source commit and passed."""
        (run,) = self.record["runs"]
        lines = self.own(run["log"]).splitlines()
        self.assertEqual(lines[0], f"== source {self.record['source_commit']} dirty=0")
        self.assertIn(f"== source after run {self.record['source_commit']} dirty=0", lines)
        self.assertEqual(lines[-1], f"== exit={run['exit']}")
        self.assertEqual((run["exit"], run["result"], run["expected"]), (0, "pass", "pass"))
        self.assertTrue(any(line.startswith("test result: ok. 1 passed;") for line in lines))
        self.assertFalse(run["heavy_model_lock"])
        self.assertTrue(any(line.startswith("== env TRANSFORMERS_NEMOTRON_H_F16_GRAPHS_DIR=") for line in lines))
        self.assertFalse(any(line.startswith("== env TRANSFORMERS_NEMOTRON_H_DIR=") for line in lines),
                         "the bundle-only run must not name the checkpoint")

    def test_source_validation_covers_the_standard_checks(self):
        """Every standard check ran from the clean worktree at the source commit, and the record states its exit."""
        validation = self.record["source_validation"]
        self.assertEqual(validation["commit"], self.record["source_commit"])
        self.assertEqual({check["name"] for check in validation["checks"]}, REQUIRED_CHECKS)
        for check in validation["checks"]:
            with self.subTest(check=check["name"]):
                body = self.own(check["log"])
                lines = body.splitlines()
                self.assertEqual(lines[0], f"== source {check['commit']} dirty=0")
                self.assertEqual(check["commit"], self.record["source_commit"])
                self.assertEqual(lines[1], f"== cmd {check['command']}")
                self.assertIn(f"== source after run {check['commit']} dirty=0", lines)
                self.assertEqual(lines[-1], f"== exit={check['exit']}")
                if "rust_tests_failed" in check:
                    self.assertEqual(check["rust_tests_failed"], 0)
                    self.assertGreater(check["rust_tests_passed"], 0)
                if "gaps" in check:
                    self.assertEqual(check["gaps"], 0)
        self.assertEqual(validation["passed"], all(check["exit"] == 0 for check in validation["checks"]))
        commands = {check["name"]: check["command"] for check in validation["checks"]}
        self.assertIn("--all-features -- -D warnings", commands["clippy-1.99.1-all-features"])
        self.assertTrue(commands["clippy-1.99.1-backends"].startswith("cargo +1.99.1 clippy"))
        self.assertIn("std,flex,cpu,cuda,rocm,tokenizers,chat-templates,autodiff,train", commands["clippy-1.99.1-backends"])
        self.assertTrue(commands["msrv-1.92.0-check"].startswith("cargo +1.92.0 check --locked --all-targets"))
        self.assertIn("--document-private-items", commands["rustdoc-private"])
        rustdoc = next(check for check in validation["checks"] if check["name"] == "rustdoc-private")
        self.assertIn("== env RUSTDOCFLAGS=-D warnings", self.own(rustdoc["log"]).splitlines())
        self.assertIn("--standards", commands["documentation-audit"])

    def test_historical_records_keep_their_bytes(self):
        """The envelope record and the records it extends keep their published bytes."""
        self.assertEqual(set(self.record["historical_records"]), HISTORICAL)
        for name, digest in self.record["historical_records"].items():
            with self.subTest(record=name):
                self.assertEqual(sha256((MEASUREMENTS / name).read_bytes()), digest)
        for name, digest in self.envelope["historical_records"].items():
            with self.subTest(earlier=name):
                self.assertEqual(self.record["historical_records"][name], digest)


if __name__ == "__main__":
    unittest.main()
