"""Validate the October 17 official half-precision calibration record without checkpoints or a GPU.

The record is published with the crate; its logs and runner scripts live in a
repository-only campaign directory, so every read of them goes through
:func:`repository_evidence.repository_evidence`. In the extracted package those
tests skip; in a Git checkout a missing file still fails them.
"""

import functools
import hashlib
import json
from pathlib import Path
import re
import unittest

from repository_evidence import repository_evidence


ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = ROOT / "acceptance/measurements"
RECORD = MEASUREMENTS / "half-precision-calibration-20261017.json"
#: The Rust module whose rule the record measured.
RULE = ROOT / "tests/support/half_precision_calibration.rs"
#: One calibration line per prompt, as `Calibration::report` prints it.
PROMPT = re.compile(
    r"prompt (\d+): (\d+) tokens, official BF16 max \|Δ\| = (\S+) at position (\d+), token (\d+), "
    r"mean \|Δ\| = (\S+); bounds max (\S+), mean (\S+)"
)
#: The greedy summary line `Calibration::report` prints after the prompts.
GREEDY = re.compile(r"greedy: threshold (\S+); (\d+) of (\d+) steps decisive \[([0-9, ]*)\]")
#: Provenance labels the Rust reader prints for each bundle dtype.
DTYPE_LABELS = {"float32": "Float32", "float64": "Float64", "bfloat16": "BFloat16"}


class HalfPrecisionCalibrationRecordTests(unittest.TestCase):
    """Tie every summarized figure to the committed logs and keep the record non-promoting."""

    @classmethod
    def setUpClass(cls):
        """Read only the published record; its logs are read on demand."""
        cls.record = json.loads(RECORD.read_text())

    def evidence(self, relative):
        """Returns the bytes of a repository-only campaign file."""
        return repository_evidence(MEASUREMENTS / relative).read_bytes()

    @functools.cached_property
    def generation(self):
        """The generation log."""
        return self.evidence(self.record["sources"][0]["log"]).decode()

    @functools.cached_property
    def calibration(self):
        """The calibration log."""
        return self.evidence(self.record["sources"][1]["log"]).decode()

    def test_record_is_evidence_not_a_support_declaration(self):
        """The record names clean sources and advertises nothing."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertFalse(record["advertised_supported"])
        self.assertTrue(record["passed"])
        self.assertRegex(record["cargo_lock_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(record["sources"][-1]["commit"], record["source_commit"])
        for source in record["sources"]:
            with self.subTest(source=source["commit"]):
                self.assertRegex(source["commit"], r"^[0-9a-f]{40}$")
                log = self.evidence(source["log"]).decode()
                self.assertTrue(log.startswith(f"== source {source['commit']} dirty=0\n"))
        self.assertTrue(record["not_run"], "the Burn half-precision gates have not run")

    def test_f16_runs_are_bounded_only_as_an_upper_limit(self):
        """The record says its BF16 bounds do not calibrate F16 runs, for which its generator wrote no bundle.

        That generator is historical: the current one writes ``float16``
        graphs, from which F16 runs are now calibrated on their own (see
        ``tests/support/half_precision_calibration.rs``), so this record's
        upper-limit note describes its own campaign only.
        """
        notes = [note for note in self.record["notes"] if "F16" in note and "upper limit" in note]
        self.assertEqual(len(notes), 1)
        self.assertIn("do not reflect F16 rounding", notes[0])
        self.assertIn("The generator writes no float16 bundle", notes[0])
        generator = (ROOT / "scripts/generate_causal_lm_reference.py").read_text()
        dtypes = re.search(r"^DTYPES = \(([^)]*)\)", generator, re.MULTILINE)
        self.assertIsNotNone(dtypes)
        self.assertIn('"float16"', dtypes.group(1), "the current generator writes the F16 graphs that calibrate F16 runs")

    def test_logs_and_runner_scripts_match_their_digests(self):
        """Every committed evidence file is authenticated by its recorded SHA-256."""
        files = {**self.record["logs"], **self.record["runner_scripts"]}
        self.assertEqual(set(self.record["logs"]), {source["log"] for source in self.record["sources"]})
        for relative, digest in files.items():
            with self.subTest(file=relative):
                self.assertEqual(hashlib.sha256(self.evidence(relative)).hexdigest(), digest)

    def test_every_process_succeeded(self):
        """Each generation and calibration process exited 0 and each calibration test passed."""
        for log in (self.generation, self.calibration):
            exits = re.findall(r"(?m)^== exit=(-?\d+)", log)
            self.assertTrue(exits)
            self.assertEqual(set(exits), {"0"})
        self.assertTrue(self.generation.rstrip().endswith("== driver exit=0"))
        self.assertEqual(self.calibration.count("test result: ok. 1 passed; 0 failed"), 6)

    def test_rule_is_the_one_its_source_applied(self):
        """The recorded rule is self-consistent and names the module that applied it at its source commit.

        The rule is historical: ``tests/support/half_precision_calibration.rs``
        now applies the official-graph envelope (its slacks are tied to the
        module by ``test_half_precision_envelope_record.py``), so this record's
        single slack is checked against its own derivation, not against the
        current module. The provisional self-consistency tolerances it
        recorded are still the module's.
        """
        rule = self.record["rule"]
        self.assertEqual(rule["module"], RULE.relative_to(ROOT).as_posix())
        self.assertEqual(rule["slack"], 1.2)
        self.assertIn("1.1725", rule["slack_basis"])
        self.assertIn("rounded up to two significant digits", rule["slack_basis"])
        source = RULE.read_text()
        self.assertNotIn("pub(crate) const SLACK:", source, "the single-slack rule is retired")
        table = re.search(r"SELF_CONSISTENCY_TOLERANCE: \[\(TensorDType, f64\); 2\] =\s*\[([^\]]*)\]", source)
        self.assertIsNotNone(table)
        entries = dict(re.findall(r"\(TensorDType::(\w+), ([0-9.]+)\)", table.group(1)))
        self.assertEqual({dtype: float(value) for dtype, value in entries.items()}, rule["self_consistency"])

    def test_bundles_were_generated_from_their_twins(self):
        """Each bundle's identity, size, and resources appear in the generation log."""
        for bundle in self.record["bundles"]:
            name = Path(bundle["bundle"]).name
            with self.subTest(bundle=name):
                start = self.generation.index(f"== {name}: official BF16 twin of {bundle['twin']} ")
                block = self.generation[start:self.generation.index("token IDs identical", start)]
                self.assertIn(f"(reference.json SHA-256 {bundle['twin_reference_json_sha256']})", block)
                self.assertIn(
                    f"== {bundle['bundle']}: reference.json SHA-256 {bundle['reference_json_sha256']}; "
                    "token IDs identical to the twin: True; dtype bfloat16",
                    self.generation,
                )
                self.assertIn(f"--revision {bundle['revision']} --dtype bfloat16", block)
                self.assertEqual(float(re.search(r"([\d.]+) real", block).group(1)), bundle["wall_seconds"])
                self.assertIn(f"{bundle['maximum_resident_set_bytes']}  maximum resident set size", block)
                self.assertIn(f"{bundle['peak_memory_footprint_bytes']}  peak memory footprint", block)

    def test_calibration_figures_match_the_log(self):
        """Every per-prompt figure, bound, and greedy set is the one the calibration test printed."""
        slack = self.record["rule"]["slack"]
        for bundle in self.record["bundles"]:
            with self.subTest(bundle=Path(bundle["bundle"]).name):
                twin = f"{bundle['twin']} ({DTYPE_LABELS[bundle['twin_dtype']]}): reference.json SHA-256 "
                self.assertIn(twin + bundle["twin_reference_json_sha256"], self.calibration)
                header = f"{bundle['bundle']} (BFloat16): reference.json SHA-256 {bundle['reference_json_sha256']}"
                start = self.calibration.index(header)
                section = self.calibration[start:self.calibration.index("greedy step 0:", start)]
                printed = [match.groups() for match in PROMPT.finditer(section)]
                self.assertEqual(len(printed), len(bundle["prompts"]))
                for index, (line, prompt) in enumerate(zip(printed, bundle["prompts"])):
                    self.assertEqual(int(line[0]), index)
                    self.assertEqual(
                        (int(line[1]), float(line[2]), int(line[3]), int(line[4]), float(line[5]), float(line[6]), float(line[7])),
                        (
                            prompt["tokens"],
                            prompt["official_max_abs_delta"],
                            prompt["max_position"],
                            prompt["max_token"],
                            prompt["official_mean_abs_delta"],
                            prompt["bound_max"],
                            prompt["bound_mean"],
                        ),
                    )
                    self.assertAlmostEqual(prompt["bound_max"], slack * prompt["official_max_abs_delta"], delta=1e-12 * prompt["bound_max"])
                    self.assertAlmostEqual(prompt["bound_mean"], slack * prompt["official_mean_abs_delta"], delta=1e-12 * prompt["bound_mean"])
                threshold, decisive, steps, listed = GREEDY.search(section).groups()
                greedy = bundle["greedy"]
                self.assertEqual(float(threshold), greedy["threshold"])
                self.assertAlmostEqual(greedy["threshold"], 2 * bundle["prompts"][0]["bound_max"], delta=1e-12 * greedy["threshold"])
                self.assertEqual(int(steps), greedy["steps"])
                self.assertEqual([int(step) for step in listed.split(",") if step.strip()], greedy["decisive_steps"])
                self.assertEqual(int(decisive), len(greedy["decisive_steps"]))


if __name__ == "__main__":
    unittest.main()
