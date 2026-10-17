"""Validate the October 17 Mac validation record without checkpoints or a GPU.

The record is published with the crate; its per-process logs, diagnostics,
summaries and copied harness live in the repository-only campaign directory
``acceptance/measurements/mac-validation-20261017/``, so every read of them
goes through :func:`repository_evidence.repository_evidence`. In the extracted
package those tests skip; in a Git checkout a missing file still fails them.
The earlier records it compares against are read the same way: their JSON
summaries directly (they are published), their logs as repository evidence.
"""

import functools
import hashlib
import json
from pathlib import Path
import re
import unittest

from repository_evidence import repository_evidence


ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE = ROOT / "acceptance"
MEASUREMENTS = ACCEPTANCE / "measurements"
RECORD = MEASUREMENTS / "mac-validation-20261017.json"
#: The f4c56c4 replay whose F32 lines and peaks this round repeats.
REPLAY = MEASUREMENTS / "real-checkpoint-replay-20261016.json"
#: The large-checkpoint replay whose quantized digests and peaks this round repeats.
LARGE_REPLAY = MEASUREMENTS / "large-checkpoint-replay-20261016.json"
#: The calibration record that named the half-precision bundles.
CALIBRATION = MEASUREMENTS / "half-precision-calibration-20261017.json"
#: Output lines that time a load and so legitimately differ between runs of the same gate.
TIMING = re.compile(r"^(loaded \d+ blocks as F32 in |FLAN-T5-small (Flex|Metal)/F32: load=)")
#: Provenance labels the Rust bundle reader prints for each manifest dtype.
DTYPE_LABELS = {"float32": "Float32", "float64": "Float64", "bfloat16": "BFloat16"}


def sha256(data):
    """Hexadecimal SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


class MacValidationRecordTests(unittest.TestCase):
    """Tie every run, quote, identity and digest of the record to its committed evidence."""

    @classmethod
    def setUpClass(cls):
        """Read only the published records; campaign files are read on demand."""
        cls.record = json.loads(RECORD.read_text())
        cls.runs = cls.record["runs"]
        cls.sources = {source["commit"] for source in cls.record["sources"]}
        cls.binaries = {binary["sha256"]: binary for binary in cls.record["binaries"]}

    def evidence(self, relative):
        """Bytes of a repository-only file named relative to ``acceptance/measurements``."""
        return repository_evidence(MEASUREMENTS / relative).read_bytes()

    def require_campaign(self):
        """Skips the whole calling test in the published crate, which omits the campaign directory."""
        repository_evidence(MEASUREMENTS / self.runs[0]["log"])

    @functools.lru_cache(maxsize=None)
    def text(self, relative):
        """Text of a repository-only file named relative to ``acceptance/measurements``."""
        return self.evidence(relative).decode()

    @functools.cached_property
    def replay_log(self):
        """The f4c56c4 replay log, authenticated by its published record's digest."""
        replay = json.loads(REPLAY.read_text())
        data = self.evidence(replay["log"])
        self.assertEqual(sha256(data), replay["log_sha256"])
        return data.decode()

    def replay_block(self, section, test):
        """Output lines and peak footprint of ``test`` in one section of the f4c56c4 replay log."""
        lines = self.replay_log.splitlines()
        start = lines.index(section)
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(("== ", "-- "))), len(lines))
        body = lines[start + 1:end]
        first = next(i for i, line in enumerate(body) if line.startswith(f"test {test} ... "))
        block = [body[first][len(f"test {test} ... "):]]
        index = first + 1
        while not body[index].startswith("test result: "):
            block.append(body[index])
            index += 1
        peak = re.fullmatch(r"\s*(\d+)  peak memory footprint", body[index + 1])
        self.assertIsNotNone(peak, f"{section} {test}")
        return block, int(peak.group(1))

    def test_record_is_evidence_not_a_support_declaration(self):
        """The record names clean sources, advertises nothing, and fails as its gates do."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["scope"], "mac_validation_round_not_a_support_declaration")
        self.assertIs(record["advertised_supported"], False)
        self.assertRegex(record["cargo_lock_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn(record["source_commit"], self.sources)
        for commit in self.sources:
            self.assertRegex(commit, r"^[0-9a-f]{40}$")
        self.assertLessEqual({run["source_commit"] for run in self.runs}, self.sources)
        gates = [run for run in self.runs if run["role"] == "gate"]
        self.assertEqual(record["passed"], all(run["status"] == "pass" for run in gates))
        self.assertFalse(record["passed"], "this round has failing gates")
        self.assertTrue(record["not_run"])
        for entry in record["not_run"]:
            self.assertTrue(entry["gates"] and entry["reason"])

    def test_every_gate_log_names_its_clean_source_binary_and_test(self):
        """Each process log names one clean source, its binary by digest, its exact test and its inputs."""
        self.require_campaign()
        for run in self.runs:
            with self.subTest(run=run["name"]):
                log = self.text(run["log"])
                self.assertEqual(run["log"], f"{self.record['campaign_directory']}/{run['name']}.log")
                self.assertEqual(re.findall(r"(?m)^== source (\S+) dirty=(\d+)$", log), [(run["source_commit"], "0")])
                binary = self.binaries[run["binary_sha256"]]
                self.assertEqual(binary["source_commit"], run["source_commit"])
                self.assertRegex(log, rf"(?m)^== binary {re.escape(binary['path'])} sha256 {run['binary_sha256']}( \(.*\))?$")
                self.assertTrue(Path(binary["path"]).name.startswith(run["test_target"].split(" ", 1)[0]))
                self.assertRegex(log, rf"(?m)^== cmd .* --ignored --exact --nocapture --test-threads 1 {re.escape(run['test'])}$")
                self.assertIn(f"\n== {run['profile']} {run['test']}\n", log)
                self.assertEqual(dict(re.findall(r"(?m)^== env ([A-Z0-9_]+)=(.*)$", log)), run["env"])

    def test_pass_counts_match_the_logs(self):
        """Statuses, exit codes, failure messages and the tallies agree with libtest's own result lines."""
        tallies = {}
        for run in self.runs:
            key = (run["group"], run["source_commit"], run["role"])
            tally = tallies.setdefault(key, [0, 0])
            tally[0] += run["passed"]
            tally[1] += run["failed"]
        recorded = {(t["group"], t["source_commit"], t["role"]): [t["passed"], t["failed"]] for t in self.record["tallies"]}
        self.assertEqual(recorded, tallies)
        failing = [run for run in self.runs if run["status"] == "fail"]
        self.assertEqual([entry["run"] for entry in self.record["failed"]], [run["name"] for run in failing])
        findings = {finding["id"] for finding in self.record["findings"]}
        for entry, run in zip(self.record["failed"], failing):
            with self.subTest(failed=entry["run"]):
                self.assertEqual(entry["failure"], run["failure"])
                self.assertEqual(entry["findings"], run["diagnosis"])
                self.assertLessEqual(set(entry["findings"]), findings)
                if "superseded_by" in entry:
                    self.assertIn(entry["superseded_by"], self.sources)
                    self.assertNotEqual(entry["superseded_by"], run["source_commit"])
        self.require_campaign()
        for run in self.runs:
            with self.subTest(run=run["name"]):
                self.assertEqual(run["passed"] + run["failed"], 1)
                log = self.text(run["log"])
                if run["status"] == "pass":
                    self.assertEqual((run["passed"], run["exit"]), (1, 0))
                    self.assertIn("\ntest result: ok. 1 passed; 0 failed;", log)
                    self.assertNotIn("panicked at", log)
                else:
                    self.assertEqual((run["status"], run["failed"], run["exit"]), ("fail", 1, 101))
                    self.assertIn("\ntest result: FAILED. 0 passed; 1 failed;", log)
                    self.assertRegex(log, rf"panicked at \S+:\d+:\d+:\n{re.escape(run['failure'])}\n")
                self.assertRegex(log, rf"(?m)^== exit={run['exit']} wall=")
                self.assertNotIn('watchdog="', log)

    def test_every_quoted_measurement_appears_in_its_log(self):
        """Run measurements, resources, check quotes and finding quotes occur verbatim in their logs."""
        self.require_campaign()
        for run in self.runs:
            with self.subTest(run=run["name"]):
                log = self.text(run["log"])
                self.assertTrue(run["measurements"])
                for measurement in run["measurements"]:
                    self.assertIn(measurement, log)
                self.assertRegex(log, rf"(?m)^\s*{run['peak_memory_footprint_bytes']}  peak memory footprint$")
                self.assertRegex(log, rf"(?m)^\s*{run['maximum_resident_set_bytes']}  maximum resident set size$")
                self.assertIn(f" wall={run['wall_seconds']}s ", log)
                self.assertIn(f"== start {run['start_utc']}", log)
        for check in self.record["checks"]:
            with self.subTest(check=check["name"]):
                log = self.text(check["log"])
                for quote in check["quotes"]:
                    self.assertIn(quote, log)
        for finding in self.record["findings"]:
            for evidence in finding["evidence"]:
                with self.subTest(finding=finding["id"], log=evidence["log"]):
                    log = self.text(evidence["log"])
                    self.assertTrue(evidence["quotes"])
                    for quote in evidence["quotes"]:
                        self.assertIn(quote, log)

    def test_files_match_their_digests_and_cover_the_campaign_directory(self):
        """Every committed file is listed exactly once and authenticated by its recorded SHA-256."""
        self.require_campaign()
        listed = [(run["log"], run["log_sha256"]) for run in self.runs]
        listed += [(check["log"], check["log_sha256"]) for check in self.record["checks"]]
        for key in ("diagnostics", "auxiliary_files", "harness_files"):
            listed += [(entry["path"], entry["sha256"]) for entry in self.record[key]]
        paths = [path for path, _ in listed]
        self.assertEqual(len(paths), len(set(paths)))
        for path, digest in listed:
            with self.subTest(path=path):
                self.assertEqual(sha256(self.evidence(path)), digest)
        campaign = MEASUREMENTS / self.record["campaign_directory"]
        repository_evidence(campaign / paths[0].split("/", 1)[1])
        present = {path.relative_to(MEASUREMENTS).as_posix() for path in campaign.rglob("*") if path.is_file()}
        self.assertEqual(present, set(paths))
        for entry in self.record["auxiliary_files"] + self.record["harness_files"]:
            self.assertEqual(len(self.evidence(entry["path"])), entry["bytes"])

    def test_diagnostics_name_their_sources(self):
        """A diagnostic that records its source names exactly that clean commit; the others name none."""
        self.require_campaign()
        for entry in self.record["diagnostics"]:
            with self.subTest(path=entry["path"]):
                sources = re.findall(r"(?m)^== source (\S+) dirty=(\d+)$", self.text(entry["path"]))
                if "source_commit" in entry:
                    self.assertIn(entry["source_commit"], self.sources)
                    self.assertEqual(sources, [(entry["source_commit"], "0")])
                else:
                    self.assertEqual(sources, [])
                self.assertTrue(entry["description"])

    def test_uncommitted_tree_is_named_by_its_logs(self):
        """The pre-commit checks name their tree, whose content the record ties to a listed commit."""
        self.require_campaign()
        for tree in self.record["uncommitted_trees"]:
            self.assertIn(tree["base"], self.sources)
            self.assertIn(tree["content_equals"], self.sources)
            self.assertRegex(tree["diff_sha256_prefix"], r"^[0-9a-f]{16}$")
            self.assertIn(tree["diff_sha256_prefix"], tree["label"])
            self.assertIn(tree["base"], tree["label"])
            for log in tree["logs"]:
                with self.subTest(log=log):
                    self.assertTrue(self.text(log).startswith(tree["label"] + "\n"))
            names = {check["log"]: check for check in self.record["checks"]}
            self.assertEqual({names[log]["tree"] for log in tree["logs"]}, {"uncommitted"})

    def test_binaries_and_harness_carry_identities(self):
        """Binaries are named by digest and source; copied scripts match the digests taken at run time."""
        for binary in self.record["binaries"]:
            with self.subTest(binary=binary["sha256"]):
                self.assertRegex(binary["sha256"], r"^[0-9a-f]{64}$")
                self.assertIn(binary["source_commit"], self.sources)
                self.assertTrue(binary["retained"])
                for path in binary.get("source_files", []) + [binary.get("patch")] * ("patch" in binary):
                    self.assertTrue(any(entry["path"] == path for entry in self.record["harness_files"] + self.record["diagnostics"]))
        self.assertLessEqual({run["binary_sha256"] for run in self.runs}, set(self.binaries))
        self.require_campaign()
        for entry in self.record["harness_files"]:
            with self.subTest(harness=entry["path"]):
                if "run_time_digest" in entry:
                    taken = entry["run_time_digest"]
                    self.assertEqual(taken["sha256"], entry["sha256"])
                    self.assertIn(f"{taken['sha256']}  {taken['listed_path']}\n", self.text(taken["listing"]))
                else:
                    self.assertRegex(entry["modified_utc"], r"^2026-\d\d-\d\dT\d\d:\d\d:\d\dZ$")

    def test_f32_reruns_repeat_the_f4c56c4_replay(self):
        """Every F32 gate repeats the f4c56c4 deviation and greedy lines; its old peak is the replay's."""
        self.require_campaign()
        for run in self.runs:
            if run["precision"] != "F32" or run["role"] not in ("gate", "repeat", "control"):
                continue
            with self.subTest(run=run["name"]):
                old = run["f4c56c4"]
                self.assertEqual(old["record"], REPLAY.name)
                block, peak = self.replay_block(old["section"], run["test"])
                self.assertEqual(old["peak_memory_footprint_bytes"], peak)
                block = [line for line in block if not line.startswith("test result: ") and not TIMING.match(line)]
                current = [line for line in run["measurements"] if not line.startswith("test result: ") and not TIMING.match(line)]
                self.assertEqual([line for line in current if line not in old["lines_new_since_f4c56c4"]], block)
                self.assertEqual(old["identical_line_count"], len(block))
                self.assertTrue(block)

    def test_quantized_runs_repeat_the_large_replay(self):
        """The quantized smokes and audits reproduce the large replay's digests; its peaks are quoted exactly."""
        large = json.loads(LARGE_REPLAY.read_text())
        text = LARGE_REPLAY.read_text()
        runs = {run["name"]: run for run in large["runs"]}
        compared = [run for run in self.runs if "large_checkpoint_replay" in run]
        self.assertEqual(len(compared), 6)
        for run in compared:
            old = run["large_checkpoint_replay"]
            self.assertEqual(runs[old["run"]]["peak_memory_footprint_bytes"][old["test"]], old["peak_memory_footprint_bytes"])
        self.require_campaign()
        for run in compared:
            with self.subTest(run=run["name"]):
                old = run["large_checkpoint_replay"]
                self.assertEqual(old["record"], LARGE_REPLAY.name)
                self.assertEqual(old["test"], run["test"])
                self.assertEqual(runs[old["run"]]["source_commit"], old["source_commit"])
                self.assertEqual(runs[old["run"]]["peak_memory_footprint_bytes"][old["test"]], old["peak_memory_footprint_bytes"])
                log = self.text(run["log"])
                for value in old["identical_values"]:
                    self.assertIn(value, log)
                    self.assertIn(value, text)

    def test_checkpoint_and_bundle_identities_match_earlier_records(self):
        """Checkpoint configs and bundle manifests carry the digests the earlier records pinned."""
        replay = json.loads(REPLAY.read_text())
        calibration = json.loads(CALIBRATION.read_text())
        configs, manifests = {}, {}
        for run in replay["runs"]:
            reference = run.get("reference") or {}
            if "config_sha256" in reference:
                configs.setdefault(run["repository"], set()).add(reference["config_sha256"])
                manifests[reference["bundle"]] = reference["reference_json_sha256"]
        for bundle in calibration["bundles"]:
            configs.setdefault(bundle["repository"], set()).add(bundle["config_sha256"])
            manifests[Path(bundle["bundle"]).name] = bundle["reference_json_sha256"]
            manifests[Path(bundle["twin"]).name] = bundle["twin_reference_json_sha256"]
        checkpoints = {entry["key"]: entry for entry in self.record["checkpoints"]}
        for entry in checkpoints.values():
            with self.subTest(checkpoint=entry["key"]):
                self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
                self.assertRegex(entry["config_sha256"], r"^[0-9a-f]{64}$")
                if "acceptance_record" in entry:
                    pinned = (ACCEPTANCE / entry["acceptance_record"]).read_text()
                    self.assertIn(entry["config_sha256"], pinned)
                    self.assertIn(entry["revision"], pinned)
                else:
                    self.assertEqual(configs[entry["repository"]], {entry["config_sha256"]})
        bundles = {entry["path"]: entry for entry in self.record["bundles"]}
        for entry in bundles.values():
            with self.subTest(bundle=entry["path"]):
                self.assertIn(entry["checkpoint"], checkpoints)
                name = Path(entry["path"]).name
                if name in manifests:
                    self.assertEqual(manifests[name], entry["manifest_sha256"])
                else:
                    pinned = (ACCEPTANCE / checkpoints[entry["checkpoint"]]["acceptance_record"]).read_text()
                    self.assertIn(entry["manifest_sha256"], pinned)
        used = set()
        for run in self.runs:
            with self.subTest(run=run["name"]):
                if run["checkpoint"] is not None:
                    self.assertIn(checkpoints[run["checkpoint"]]["path"], run["env"].values())
                for key, value in run["env"].items():
                    if key.endswith("REFERENCE_DIR"):
                        self.assertEqual(bundles[value]["checkpoint"], run["checkpoint"])
                        used.add(value)
        self.assertEqual(used, set(bundles))
        self.require_campaign()
        for run in self.runs:
            if run["role"] == "calibration":
                log = self.text(run["log"])
                for value in run["env"].values():
                    if value in bundles:
                        label = DTYPE_LABELS[bundles[value]["dtype"]]
                        self.assertIn(f"{value} ({label}): reference.json SHA-256 {bundles[value]['manifest_sha256']}", log)

    def test_burnpacks_are_byte_identical_to_the_authenticated_ones(self):
        """Each new Burnpack's size and digest equal the authenticated copy's and the checkpoint record's."""
        runs = {run["name"]: run for run in self.runs}
        checkpoints = {entry["key"]: entry for entry in self.record["checkpoints"]}
        self.assertEqual(len(self.record["burnpacks"]), 4)
        for burnpack in self.record["burnpacks"]:
            with self.subTest(burnpack=burnpack["path"]):
                run = runs[burnpack["created_by"]]
                self.assertEqual(run["status"], "pass")
                self.assertIn(burnpack["path"], run["env"].values())
                size, digest = burnpack["bytes"], burnpack["sha256"]
                self.assertIn(f"== artifact {burnpack['path']} bytes={size} sha256={digest} ", run["measurements"][-2] + " ")
                self.assertEqual(
                    run["measurements"][-1],
                    f"== authenticated {burnpack['authenticated_copy']} bytes={size} sha256={digest} identical=True",
                )
                pinned = (ACCEPTANCE / checkpoints[burnpack["checkpoint"]]["acceptance_record"]).read_text()
                self.assertIn(digest, pinned)
                users = [other for other in self.runs if burnpack["path"] in other["env"].values() and other is not run]
                self.assertTrue(users and all(other["status"] == "pass" for other in users))

    def test_group_summaries_cover_every_run(self):
        """Each run belongs to a summarized group, and each group names the sources its runs used."""
        groups = {group["name"]: group for group in self.record["groups"]}
        for run in self.runs:
            with self.subTest(run=run["name"]):
                self.assertIn(run["group"], groups)
                self.assertIn(run["source_commit"], groups[run["group"]]["source_commits"])
                self.assertTrue(run["log"].startswith(groups[run["group"]]["directory"] + "/"))
        self.assertEqual({run["group"] for run in self.runs}, set(groups))


if __name__ == "__main__":
    unittest.main()
