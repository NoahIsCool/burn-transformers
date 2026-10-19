"""Validate the October 18 Nemotron-H envelope record without checkpoints, bundles, or a GPU.

The record is published with the crate. Its method, derivation, copied
bundle manifests, generator and gate logs, and source-validation logs live in
the repository-only campaign directory
``acceptance/measurements/nemotron-h-envelope-20261018/``, so every read of
them goes through :func:`repository_evidence.repository_evidence`; in the
extracted package those tests skip, and in a Git checkout a missing file
fails them. The tests tie every figure, quote, digest, and status of the
record to its evidence; re-derive the envelope, the leave-one-out ratios, the
rounded slacks, and the cosine floors from the derivation's per-graph worst
rows; require the method, the derivation, and its assessment to precede every
Burn run; tie the compiled Rust constants (slacks, bundle digests, generator
digest) to the evidence; and require the earlier records to keep their bytes.

The maintainer did not adopt the envelope rule (decision of 2026-10-18, see
``nemotron-h-envelope-decision-20261018.json``), so the compiled slacks were
removed after this record's code head; the test reads them from the code
head's source in Git, which the published crate (without Git history) skips.
"""

import datetime as dt
import functools
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import unittest

from repository_evidence import PACKAGE_CONTENTS, repository_evidence


ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = ROOT / "acceptance/measurements"
RECORD = MEASUREMENTS / "nemotron-h-envelope-20261018.json"
ENVELOPE_ROUND = MEASUREMENTS / "half-precision-envelope-20261018.json"
PARITY = "tests/support/nemotron_h_parity.rs"
GRAPH_READER = ROOT / "tests/support/nemotron_h_f16_graphs.rs"
GENERATOR = ROOT / "scripts/generate_nemotron_h_f16_graph_reference.py"
SPEC = importlib.util.spec_from_file_location("derive_half_precision_envelope",
                                              ROOT / "scripts/derive_half_precision_envelope.py")
ROUNDING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUNDING)
#: Records the round must leave untouched.
HISTORICAL = frozenset({
    "half-precision-calibration-20261017.json", "mac-validation-20261017.json",
    "half-precision-envelope-20261018.json", "half-precision-followups-20261017.json",
    "half-precision-review-20261018.json", "reference-precision-followups-20261016.json",
    "../nemotron-3.5-lightning-30b-a3b-a9904d24bcc1d289a1950fa9d2b978c47cf903b9.json",
})
#: Row paths in gate order.
PATHS = ("prompt", "teacher", "cached")


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


def source_at(commit, relative):
    """Text of the repository file ``relative`` at ``commit``, read from Git.

    The published crate carries no Git history, so there the calling test is
    skipped; in a Git checkout a missing commit or path fails it.
    """
    if PACKAGE_CONTENTS.is_packaged_archive(ROOT):
        raise unittest.SkipTest(f"{relative} at {commit} needs the Git history of the repository")
    shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
                           capture_output=True, text=True, check=False)
    if shown.returncode:
        raise AssertionError(f"git show {commit}:{relative} failed: {shown.stderr.strip()}")
    return shown.stdout


def utc(stamp):
    """Parses an ISO 8601 UTC time with a ``Z`` or ``+00:00`` suffix."""
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


class NemotronEnvelopeRecordTests(unittest.TestCase):
    """Tie every status, figure, quote, ordering, and identity of the record to its committed evidence."""

    @classmethod
    def setUpClass(cls):
        """Read only the published record; campaign files are read on demand."""
        cls.record = json.loads(RECORD.read_text())
        cls.campaign = cls.record["campaign_directory"]

    def evidence(self, relative):
        """Bytes of a campaign file, checked against ``files``."""
        data = repository_evidence(MEASUREMENTS / self.campaign / relative).read_bytes()
        self.assertEqual(sha256(data), self.record["files"][relative], relative)
        return data

    @functools.lru_cache(maxsize=None)
    def text(self, relative):
        """Text of an authenticated campaign file."""
        return self.evidence(relative).decode()

    def campaign_json(self, relative):
        """An authenticated campaign JSON file."""
        return json.loads(self.text(relative))

    def test_record_is_evidence_not_a_support_declaration(self):
        """The record promotes nothing, passes only with every run, and names what it did not run."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertFalse(record["advertised_supported"])
        self.assertTrue(record["scope"].endswith("not_a_support_declaration"))
        self.assertEqual(record["passed"], all(run["result"] == "pass" for run in record["runs"]))
        for key in ("base_commit", "source_commit", "code_head", "documentation_commit"):
            self.assertRegex(record[key], r"^[0-9a-f]{40}$")
        self.assertTrue(record["not_run"])
        self.assertTrue(any("Adopting the four-graph envelope rule" in item for item in record["awaiting_maintainer"]))
        self.assertEqual([decision["user_review"] for decision in record["decisions"]], [True])

    def test_files_cover_the_campaign_directory(self):
        """``files`` lists exactly the campaign directory's files, each with its digest."""
        root = repository_evidence(MEASUREMENTS / self.campaign / "derivation/method.txt").parents[1]
        present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        self.assertEqual(present, set(self.record["files"]))
        for relative in sorted(present):
            with self.subTest(file=relative):
                self.evidence(relative)

    def test_every_quote_is_in_its_log(self):
        """Every quote anywhere in the record is a line of the log it cites, or part of one."""
        for entry in quoted_entries(self.record):
            lines = self.text(entry["log"]).splitlines()
            for quote in entry["quotes"]:
                with self.subTest(log=entry["log"], quote=quote[:60]):
                    self.assertGreaterEqual(len(quote), 20)
                    self.assertTrue(any(quote in line for line in lines), quote)

    def test_method_and_derivation_precede_every_burn_run(self):
        """The method precedes the new graphs; the derivation and its assessment precede every Burn run."""
        method = self.text("derivation/method.txt")
        registered = utc(re.search(r"Registered (\S+Z) at source ([0-9a-f]{40})", method).group(1))
        self.assertEqual(registered, utc(self.record["method"]["registered_utc"]))
        self.assertIn("before any new official evaluation was\ngenerated and before any Burn row", method)
        tables = self.text("derivation/envelope.txt")
        self.assertIn("== burn rows computed in this round before this derivation: none", tables.splitlines())
        self.assertIn("== exit=0", tables.splitlines())
        derived = utc(self.record["derivation"]["derived_utc"])
        assessed = utc(self.record["derivation"]["assessed_utc"])
        self.assertIn(f"Written {self.record['derivation']['assessed_utc']}", self.text("derivation/assessment.txt"))
        for label in ("cpu-sdpa", "mps-sdpa", "mps-eager-control"):
            start = re.search(r"^== start (\S+)$", self.text(f"generator/logs/{label}.log"), re.M).group(1)
            self.assertLess(registered, utc(start), label)
            self.assertLess(utc(start), derived, label)
        self.assertLess(derived, assessed)
        for run in self.record["runs"]:
            start = re.search(r"^== start (\S+)$", self.text(run["log"]), re.M).group(1)
            self.assertLess(assessed, utc(start), run["name"])

    def test_envelope_slacks_and_floors_follow_from_the_official_worst_rows(self):
        """Each path's envelope, leave-one-out ratios, rounded slack, bound, and floor recompute from the graphs."""
        derivation = self.campaign_json("derivation/envelope.json")
        self.assertEqual(derivation["graph_count"], 4)
        self.assertEqual(derivation["graphs"], ["cpu-eager", "cpu-sdpa", "mps-eager", "mps-sdpa"])
        recorded = self.record["derivation"]
        for kind in PATHS:
            with self.subTest(path=kind):
                path = derivation["paths"][kind]
                maxima = {name: graph["max"] for name, graph in path["graphs"].items()}
                for name, graph in path["graphs"].items():
                    worst = max(math.sqrt(2.0 * max(0.0, 1.0 - cosine)) for cosine in graph["cosines"])
                    self.assertEqual(worst, graph["max"], name)
                    self.assertEqual(min(graph["cosines"]), graph["minimum_cosine"])
                envelope = max(maxima.values())
                ratios = {name: value / max(v for n, v in maxima.items() if n != name) for name, value in maxima.items()}
                self.assertEqual(ratios, path["leave_one_out"])
                slack = ROUNDING.round_up(max(ratios.values()))
                self.assertEqual(slack, 1.1)
                self.assertEqual((envelope, slack, slack * envelope), (path["envelope"], path["slack"], path["bound"]))
                self.assertEqual(1.0 - (slack * envelope) * (slack * envelope) / 2.0, path["cosine_floor"])
                self.assertEqual(recorded["slacks"][kind], slack)
                self.assertEqual(recorded["cosine_floor"][kind], path["cosine_floor"])
                self.assertLess(recorded["cosine_floor"][kind], recorded["previous_floor"][kind])
                self.assertGreaterEqual(path["cosine_floor"], 0.99)
        self.assertEqual(recorded["largest_ratio"], {k: derivation["paths"][k]["largest_ratio"] for k in PATHS})

    def test_compiled_constants_are_the_evidence(self):
        """The code head's Rust slacks, and the pinned bundle and generator digests, are the derivation's and the bundles'."""
        parity = source_at(self.record["code_head"], PARITY)
        for kind, name in (("prompt", "PROMPT_SLACK"), ("teacher", "TEACHER_SLACK"), ("cached", "CACHED_SLACK")):
            value = float(re.search(rf"pub\(super\) const {name}: f64 = ([0-9.]+);", parity).group(1))
            self.assertEqual(value, self.record["derivation"]["slacks"][kind], name)
        reader = GRAPH_READER.read_text()
        digests = self.campaign_json("bundles/bundle-digests.json")
        for key, dirname in (("f16-cpu-sdpa", "f16-cpu-sdpa.bundle"), ("f16-mps-sdpa", "f16-mps-sdpa.bundle")):
            block = reader[reader.index(f'dirname: "{dirname}"'):]
            block = block[:block.index("},")]
            files = digests[key]["files"]
            self.assertIn(f'manifest_sha256: "{files["reference.json"]["sha256"]}"', block)
            for kind in ("Prompt", "Teacher", "Cached"):
                digest = files[f"float16-{kind.lower()}-logits.f32le"]["sha256"]
                self.assertRegex(block, rf'RowKind::{kind},\s+"{digest}"')
            self.assertEqual(sha256(self.evidence(f"bundles/{key}/reference.json")), files["reference.json"]["sha256"])
        generator = re.search(r'const GENERATOR_SHA256: &str = "([0-9a-f]{64})"', reader).group(1)
        self.assertEqual(sha256(GENERATOR.read_bytes()), generator)
        for graph in self.record["graphs"]:
            if "generator_sha256" in graph:
                self.assertEqual(graph["generator_sha256"], generator)

    def test_new_graph_manifests_are_the_registered_graphs(self):
        """Each new bundle evaluates its graph in F16 with the self-check passed and offload on the device."""
        derivation = self.campaign_json("derivation/envelope.json")
        for key, device, experts in (("f16-cpu-sdpa", "cpu", "grouped_mm"), ("f16-mps-sdpa", "mps", "eager"),
                                     ("f16-mps-eager-control", "mps", "eager")):
            with self.subTest(bundle=key):
                manifest = self.campaign_json(f"bundles/{key}/reference.json")
                attention = "eager" if key.endswith("control") else "sdpa"
                self.assertEqual(manifest["graph"]["resolved"],
                                 {"attention_implementation": attention, "experts_implementation": experts})
                self.assertEqual((manifest["graph"]["dtype"], manifest["graph"]["device"]), ("float16", device))
                provenance = manifest["provenance"]
                self.assertEqual(provenance["self_check"]["harness_bitwise_equal"], True)
                self.assertEqual(provenance["self_check"]["offloaded_bitwise_equal"], True)
                self.assertEqual(provenance["offload_execution_devices"], [device])
                self.assertEqual((provenance["max_resident_gib"], provenance["offloaded_blocks"]), (16.0, 23))
                self.assertEqual(provenance["environment"]["PYTORCH_ENABLE_MPS_FALLBACK"], "0")
                for phase in provenance["timings"]:
                    sample = phase.get("host") or phase["before"]
                    self.assertEqual(len(sample["load_average"]), 3)
                    self.assertTrue(sample["top_cpu"])
                if not key.endswith("control"):
                    graph = f"{device}-{attention}"
                    for kind in PATHS:
                        self.assertAlmostEqual(manifest["comparisons"]["float32"][kind]["minimum"],
                                               derivation["paths"][kind]["graphs"][graph]["minimum_cosine"], places=10)

    def test_generator_runs_have_their_recorded_outcomes_and_host_samples(self):
        """Each graph log names the export commit and tool, samples host load, and ends with its manifest digest."""
        runs = [json.loads(raw) for raw in self.text("generator/logs/runs.jsonl").splitlines()]
        self.assertEqual([run["label"] for run in runs], ["cpu-sdpa", "mps-eager-control", "mps-sdpa"])
        commits = {graph["generator_commit"] for graph in self.record["graphs"] if "generator_commit" in graph}
        self.assertEqual(len(commits), 1)
        digests = self.campaign_json("bundles/bundle-digests.json")
        for run in runs:
            with self.subTest(run=run["label"]):
                text = self.text(f"generator/logs/{run['label']}.log")
                lines = text.splitlines()
                self.assertEqual(lines[0].split()[:3], ["==", "source", *commits])
                self.assertTrue(any(line.startswith("== start uptime ") for line in lines))
                self.assertTrue(any(line.startswith("== start top-cpu ") for line in lines))
                self.assertTrue(any(line.startswith("== end top-cpu ") for line in lines))
                self.assertIn("== exit=0 ", text)
                self.assertNotIn("WATCHDOG_KILLED", text)
                self.assertEqual(run["status"], 0)
                manifest = re.search(r"^== reference.json sha256 ([0-9a-f]{64})$", text, re.M).group(1)
                self.assertEqual(manifest, run["reference_json_sha256"])
                self.assertEqual(manifest, digests[f"f16-{run['label']}"]["files"]["reference.json"]["sha256"])
        attempt = [json.loads(raw) for raw in self.text("generator/logs/attempt-1/runs.jsonl").splitlines()]
        self.assertEqual([(run["label"], run["status"]) for run in attempt], [("mps-sdpa", -9), ("cpu-sdpa", -9)])
        self.assertEqual([run["killed"] for run in self.record["attempts"][0]["runs"]], [run["killed"] for run in attempt])

    def test_burn_runs_have_their_recorded_status_source_and_binary(self):
        """Each Burn log names the clean source commit and a recorded binary, and passes."""
        binaries = {binary["path"]: binary["sha256"] for binary in self.record["binaries"]}
        build = self.text("gates/logs/build.log").splitlines()
        self.assertIn(f"== source {self.record['source_commit']} dirty=0", build)
        self.assertIn(f"== source after build {self.record['source_commit']} dirty=0", build)
        jsonl = [json.loads(raw) for raw in self.text("gates/logs/runs.jsonl").splitlines()]
        self.assertEqual([run["name"] for run in jsonl], [run["name"] for run in self.record["runs"]])
        for run in self.record["runs"]:
            with self.subTest(run=run["name"]):
                text = self.text(run["log"])
                lines = text.splitlines()
                self.assertEqual(lines[0], f"== source {self.record['source_commit']} dirty=0")
                self.assertIn(f"== source after run {self.record['source_commit']} dirty=0", lines)
                path, digest = re.search(r"^== binary (.+) sha256 ([0-9a-f]{64})$", text, re.M).groups()
                self.assertEqual(binaries[path], digest)
                self.assertEqual(int(re.search(r"^== exit=(-?\d+)", text, re.M).group(1)), run["exit"])
                self.assertEqual("pass" if "test result: ok. 1 passed" in text else "fail", run["result"])
                self.assertEqual(run["result"], run["expected"])
                self.assertNotIn("WATCHDOG_KILLED", text)
                self.assertTrue(any(line.startswith("== start top-cpu ") for line in lines))
        self.assertEqual(self.record["tallies"]["as_expected"], len(self.record["runs"]))
        driver = self.text("gates/logs/driver.log")
        self.assertIn("lock acquired for nemotron-h.f16.metal.official-precision", driver)
        self.assertIn("lock released", driver)

    def test_burn_rows_meet_the_derived_bounds_and_reproduce_the_failing_row(self):
        """Burn's printed rows equal the record, meet the derived bounds, and cached 22 is the earlier failing cosine."""
        derivation = self.campaign_json("derivation/envelope.json")
        log = self.text("gates/logs/nemotron-h.f16.metal.official-precision.log")
        for kind in PATHS:
            with self.subTest(path=kind):
                burn = self.record["burn"]["paths"][kind]
                path = derivation["paths"][kind]
                self.assertEqual(len(burn["cosines"]), path["rows"])
                self.assertAlmostEqual(min(burn["cosines"]), burn["minimum_cosine"], places=9)
                self.assertAlmostEqual(burn["cosine_floor"], path["cosine_floor"], places=9)
                self.assertAlmostEqual(burn["bound"], path["bound"], delta=5e-7 * path["bound"])
                delta = (2 * (1 - burn["minimum_cosine"])) ** 0.5
                self.assertAlmostEqual(delta, burn["normalized_deviation"], delta=1e-6 * delta)
                self.assertAlmostEqual(delta / path["envelope"], burn["of_envelope"], places=4)
                self.assertLessEqual(delta, path["bound"])
                self.assertTrue(all(cosine >= path["cosine_floor"] for cosine in burn["cosines"]))
                self.assertIn(f"{kind.capitalize()}: {path['rows']} rows, minimum cosine {burn['minimum_cosine']:.12f}", log)
        cached = self.record["burn"]["paths"]["cached"]
        self.assertGreater(cached["of_envelope"], max(derivation["paths"]["cached"]["leave_one_out"].values()))
        failing = next(entry for entry in json.loads(ENVELOPE_ROUND.read_text())["findings"]
                       if entry["id"] == "nemotron-floor-failure")
        earlier = float(re.search(r"cosine ([0-9.]+) against", failing["evidence"][0]["quotes"][0]).group(1))
        self.assertEqual(f"{earlier:.12f}", f"{cached['minimum_cosine']:.12f}")
        self.assertEqual(cached["decision"], 22)

    def test_control_is_reported_and_not_adopted(self):
        """The MPS eager repeat differs from the oracle, and the derivation leaves it out of the envelope."""
        derivation = self.campaign_json("derivation/envelope.json")
        (control,) = derivation["controls_reported_not_adopted"]
        recorded = self.record["controls"][0]
        self.assertFalse(recorded["adopted"])
        self.assertEqual(control["repeats"], "mps-eager")
        for kind in PATHS:
            self.assertFalse(control["paths"][kind]["bitwise_equal_to_the_repeated_graph"])
            self.assertEqual(recorded["paths"][kind]["minimum_cosine_to_the_oracle"],
                             control["paths"][kind]["minimum_cosine_to_the_repeated_graph"])
        self.assertNotIn("mps-eager-control", derivation["graphs"])
        self.assertEqual(recorded["platform"], "macOS-27.0.1-arm64-arm-64bit")

    def test_source_validation_passed(self):
        """Every source check ran from a clean tree at a round commit and exited zero."""
        validation = self.record["source_validation"]
        commits = {self.record["source_commit"], self.record["documentation_commit"]}
        for check in validation["checks"]:
            with self.subTest(check=check["name"]):
                lines = self.text(check["log"]).splitlines()
                self.assertEqual(lines[0], f"== source {check['commit']} dirty=0")
                self.assertIn(check["commit"], commits)
                self.assertEqual(lines[1], f"== cmd {check['command']}")
                self.assertEqual(lines[-1], f"== exit={check['exit']}")
                self.assertEqual(check["exit"], 0)
        self.assertEqual({check["name"] for check in validation["checks"]},
                         {"fmt", "clippy-metal", "clippy-default", "test-nemotron-default", "test-nemotron-metal",
                          "msrv-1.92.0-test-nemotron", "python-tests"})
        self.assertTrue(validation["passed"])

    def test_historical_records_keep_their_bytes(self):
        """The records this one extends keep their published bytes."""
        self.assertEqual(set(self.record["historical_records"]), HISTORICAL)
        for name, digest in self.record["historical_records"].items():
            with self.subTest(record=name):
                self.assertEqual(sha256((MEASUREMENTS / name).read_bytes()), digest)


if __name__ == "__main__":
    unittest.main()
