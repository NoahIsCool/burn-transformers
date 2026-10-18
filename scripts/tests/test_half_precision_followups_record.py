"""Validate the half-precision follow-ups record (named for the round's start on October 17) without checkpoints or a GPU.

The record is published with the crate. Its rerun logs, harness, input
identities, final head checks, and stage reports live in the repository-only
campaign directory ``acceptance/measurements/half-precision-followups-20261017/``,
and the earlier runs it is compared with live in the envelope and Mac
validation campaign directories, so every read of a log goes through
:func:`repository_evidence.repository_evidence`. In the extracted package
those tests skip; in a Git checkout a missing file fails them. Every log read
from another campaign is first authenticated by the digest that campaign's
published record gives it. The tests recompute, from the logs alone, each
run's status, the identity of its output with the envelope round's run of the
same name, and the F32 statistics it repeats from the pre-SiLU round, so a
tampered quote, status, digest, or comparison fails them.
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
RECORD = MEASUREMENTS / "half-precision-followups-20261017.json"
#: The implementation stage's record, whose runs at f9f78b5 this rerun repeats.
ENVELOPE = MEASUREMENTS / "half-precision-envelope-20261018.json"
#: The Mac validation record, whose F32 runs precede the once-rounded SiLU.
MAC = MEASUREMENTS / "mac-validation-20261017.json"
#: Records the round must leave untouched.
HISTORICAL = frozenset({
    "half-precision-calibration-20261017.json",
    "mac-validation-20261017.json",
    "half-precision-envelope-20261018.json",
})
#: Decision fields the record repeats verbatim from the decision stage's report.
DECISION_FIELDS = ("question", "decision", "rationale", "alternatives_rejected", "user_review")
#: Run-specific details masked before two runs' outputs are compared.
MASK_THREAD = re.compile(r"thread '([^']*)' \(\d+\)")
MASK_TIME = re.compile(r"finished in [\d.]+s")


def sha256(data):
    """Hexadecimal SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def body(text):
    """A log's test output from ``running 1 test`` to ``test result``, thread ids and timings masked."""
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith("running "):
            inside = True
        if inside:
            line = MASK_THREAD.sub(r"thread '\1' (TID)", line)
            lines.append(MASK_TIME.sub("finished in Xs", line))
        if line.startswith("test result"):
            break
    return lines


def exit_status(text):
    """The ``== exit=`` status a run log records."""
    return int(re.search(r"(?m)^== exit=(-?\d+)", text).group(1))


def peak_footprint(text):
    """The peak memory footprint ``/usr/bin/time -l`` printed in a run log."""
    return int(re.findall(r"(?m)^\s*(\d+)  peak memory footprint$", text)[-1])


def stat_lines(lines):
    """Statistic lines (those holding ``|Δ|``), without the first line's ``test NAME ... `` prefix."""
    return [re.sub(r"^test \S+ \.\.\. ", "", line) for line in lines if "|Δ|" in line]


def greedy_counts(lines):
    """``[label, matched, total]`` of each greedy summary line, in either round's wording."""
    out = []
    for line in lines:
        match = re.match(r"^((?:long )?greedy): (\d+) of (\d+) (?:greedy )?tokens", re.sub(r"^test \S+ \.\.\. ", "", line))
        if match:
            out.append([match.group(1), int(match.group(2)), int(match.group(3))])
    return out


class HalfPrecisionFollowupsRecordTests(unittest.TestCase):
    """Tie every run, quote, comparison, identity, and decision of the record to its committed evidence."""

    @classmethod
    def setUpClass(cls):
        """Read only the published records; campaign files are read on demand."""
        cls.record = json.loads(RECORD.read_text())
        cls.envelope = json.loads(ENVELOPE.read_text())
        cls.mac = json.loads(MAC.read_text())
        cls.runs = cls.record["runs"]
        cls.head = cls.record["source_commit"]
        cls.campaign = cls.record["campaign_directory"]
        cls.binaries = {binary["sha256"]: binary for binary in cls.record["binaries"]}

    def require_campaign(self):
        """Skips the calling test in the published crate, which omits the campaign directory."""
        repository_evidence(MEASUREMENTS / self.runs[0]["log"])

    @functools.lru_cache(maxsize=None)
    def text(self, relative):
        """Text of a repository-only file named relative to ``acceptance/measurements``, authenticated by digest.

        A file of this record's campaign must match the record's ``files``; a
        file of the envelope or Mac validation campaign must match the digest
        that campaign's published record lists for it.
        """
        data = repository_evidence(MEASUREMENTS / relative).read_bytes()
        campaign, _, inside = relative.partition("/")
        if campaign == self.campaign:
            expected = self.record["files"][relative]
        elif campaign == self.envelope["campaign_directory"]:
            expected = self.envelope["files"][inside]
        elif campaign == self.mac["campaign_directory"]:
            expected = next(run["log_sha256"] for run in self.mac["runs"] if run["log"] == relative)
        else:
            raise AssertionError(f"{relative} belongs to no record this one cites")
        self.assertEqual(sha256(data), expected, relative)
        return data.decode()

    def test_record_is_evidence_not_a_support_declaration(self):
        """The record names its sources, promotes nothing, and fails as long as a carried failure stands."""
        record = self.record
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["scope"], "half_precision_followups_round_not_a_support_declaration")
        self.assertIs(record["advertised_supported"], False)
        self.assertRegex(record["cargo_lock_sha256"], r"^[0-9a-f]{64}$")
        roles = {source["commit"]: source["role"] for source in record["sources"]}
        for commit in roles:
            self.assertRegex(commit, r"^[0-9a-f]{40}$")
        self.assertEqual(roles[self.head], "rerun head")
        self.assertEqual(record["rerun_passed"], all(run["status"] != "fail" for run in self.runs))
        self.assertEqual(record["passed"], record["rerun_passed"] and not record["failures"])
        self.assertFalse(record["passed"], "the Nemotron-H failure carried from f9f78b5 stands")
        for failure in record["failures"]:
            self.assertEqual(failure["status"], "fail")
            self.assertIs(failure["rerun"], False)
            self.assertIn(failure["source_commit"], roles)
            run = next(run for run in self.envelope["runs"] if run["name"] == failure["run"])
            self.assertEqual(run["status"], "fail")
            self.assertEqual(f"{self.envelope['campaign_directory']}/{run['log']}", failure["log"])
        self.assertTrue(record["not_run"])
        for entry in record["not_run"]:
            self.assertTrue(entry["gates"] and entry["reason"])

    def test_files_match_their_digests_and_cover_the_campaign_directory(self):
        """Every campaign file is listed once with its SHA-256, and no file is unlisted."""
        self.require_campaign()
        files = self.record["files"]
        for relative, digest in files.items():
            with self.subTest(file=relative):
                self.assertTrue(relative.startswith(self.campaign + "/"))
                self.assertEqual(sha256((MEASUREMENTS / relative).read_bytes()), digest)
        directory = MEASUREMENTS / self.campaign
        present = {path.relative_to(MEASUREMENTS).as_posix() for path in directory.rglob("*")
                   if path.is_file() and path.name != ".DS_Store"}
        self.assertEqual(present, set(files))

    def test_run_time_digests_name_the_committed_copies(self):
        """The rerun's own SHA256SUMS lists exactly the committed logs and harness, and the prior binaries' digests."""
        self.require_campaign()
        listing = self.record["run_time_digests"]
        rerun = listing["file"].rsplit("/", 1)[0]
        listed = dict((path, digest) for digest, path in re.findall(r"(?m)^([0-9a-f]{64})  (\S.*)$", self.text(listing["file"])))
        copies = {key[len(rerun) + 1:]: digest for key, digest in self.record["files"].items()
                  if key.startswith((f"{rerun}/logs/", f"{rerun}/harness/"))}
        self.assertEqual({path: digest for path, digest in listed.items() if not path.startswith("prior-binaries/")}, copies)
        prior = {path.rsplit("/", 1)[1]: digest for path, digest in listed.items() if path.startswith("prior-binaries/")}
        self.assertEqual(prior, {Path(binary["path"]).name: binary["sha256"] for binary in self.record["binaries"]})
        again = {Path(path).name: digest for digest, path in re.findall(r"(?m)^([0-9a-f]{64})  (\S.*)$", self.text(listing["prior_binaries"]))}
        self.assertEqual(again, prior)

    def test_runs_jsonl_matches_the_runs(self):
        """The runner's one line per process agrees with the record's runs and their logs' digests."""
        self.require_campaign()
        lines = [json.loads(line) for line in self.text(self.runs[0]["log"].rsplit("/", 1)[0] + "/runs.jsonl").splitlines()]
        self.assertEqual([line["name"] for line in lines], [run["name"] for run in self.runs])
        for line, run in zip(lines, self.runs):
            with self.subTest(run=run["name"]):
                self.assertEqual((line["kind"], line["test"], line["exit"]), (run["kind"], run["test"], run["exit"]))
                self.assertEqual((line["wall_seconds"], line["peak_memory_footprint_bytes"]), (run["wall_seconds"], run["peak_memory_footprint_bytes"]))
                self.assertEqual(line["log_sha256"], run["log_sha256"])
                self.assertEqual(self.record["files"][run["log"]], run["log_sha256"])
                self.assertIsNone(line["watchdog"])
                self.assertEqual(line["result"] == "pass", run["status"] == "pass")

    def test_every_run_log_names_the_clean_head_binary_test_and_inputs(self):
        """Each log opens and closes at the clean head and names its checkout, binary, test, variables, and command."""
        self.require_campaign()
        checkout = self.record["checkout"]
        for run in self.runs:
            with self.subTest(run=run["name"]):
                log = self.text(run["log"])
                self.assertEqual(run["log"], f"{self.campaign}/rerun/logs/{run['name']}.log")
                self.assertTrue(log.startswith(f"== source {self.head} dirty=0\n"))
                self.assertEqual(log.rstrip("\n").splitlines()[-1], f"== source after run {self.head} dirty=0")
                self.assertEqual(re.findall(r"(?m)^== source (\S+) dirty=(\d+)$", log), [(self.head, "0")])
                self.assertIn(f"\n== checkout {checkout['path']} (branch {checkout['branch']})\n", log)
                binary = self.binaries[run["binary_sha256"]]
                self.assertIn(f"\n== binary {binary['path']} sha256 {run['binary_sha256']}\n", log)
                self.assertIn(f"\n== {run['kind']} {run['test']}\n", log)
                self.assertEqual(dict(re.findall(r"(?m)^== env ([A-Z0-9_]+)=(.*)$", log)), run["env"])
                flag = " --ignored" if run["ignored"] else ""
                self.assertRegex(log, rf"(?m)^== cmd /usr/bin/time -l {re.escape(binary['path'])}{flag} --exact --nocapture --test-threads 1 {re.escape(run['test'])}$")
                self.assertIn(f"\n== start {run['start_utc']} ", log)
                self.assertRegex(log, rf"(?m)^== exit={run['exit']} wall={run['wall_seconds']:.1f}s ")
                self.assertEqual(peak_footprint(log), run["peak_memory_footprint_bytes"])
                self.assertRegex(log, rf"(?m)^\s*{run['maximum_resident_set_bytes']}  maximum resident set size$")
                self.assertNotIn("watchdog=", log)
                for quote in run["quotes"]:
                    self.assertIn(quote, log)

    def test_statuses_follow_the_logs(self):
        """A pass is libtest's own pass; an expected failure fails with the message its decision predicts."""
        decisions = {decision["id"] for decision in self.record["decisions"]}
        self.require_campaign()
        for run in self.runs:
            with self.subTest(run=run["name"]):
                log = self.text(run["log"])
                self.assertIn(run["status"], ("pass", "expected_fail"), "the rerun has no unexpected failure")
                if run["status"] == "pass":
                    self.assertEqual(run["exit"], 0)
                    self.assertIn("\ntest result: ok. 1 passed; 0 failed;", log)
                    self.assertNotIn("expected_failure", run)
                else:
                    self.assertEqual(run["exit"], 101)
                    self.assertIn("\ntest result: FAILED. 0 passed; 1 failed;", log)
                    self.assertIn(run["expected_failure"], log)
                    self.assertIn(run["decision"], decisions)
                    if run["decision"] == "b-f16":
                        self.assertIn("Qwen2-7B-f64.f16.", run["name"], "only Qwen2-7B is uncalibrated in F16")
                    else:
                        self.assertEqual(run["group"], "refusals", "only the refusal checks exercise decision (d)")

    def test_tallies_follow_the_runs(self):
        """Group and total tallies, and the total wall time, are the runs' own."""
        groups = {}
        for run in self.runs:
            tally = groups.setdefault(run["group"], {"pass": 0, "expected_fail": 0, "fail": 0})
            tally[run["status"]] += 1
        self.assertEqual(self.record["tallies"]["groups"], groups)
        total = {status: sum(group[status] for group in groups.values()) for status in ("pass", "expected_fail", "fail")}
        self.assertEqual(self.record["tallies"]["total"], total)
        self.assertEqual(self.record["total_wall_seconds"], round(sum(run["wall_seconds"] for run in self.runs), 1))
        largest = max(self.runs, key=lambda run: run["peak_memory_footprint_bytes"])
        self.assertEqual(self.record["largest_peak"]["run"], largest["name"])
        self.assertEqual(self.record["largest_peak"]["peak_memory_footprint_bytes"], largest["peak_memory_footprint_bytes"])

    def test_runs_repeat_the_envelope_round_at_f9f78b5(self):
        """Every run with an envelope counterpart has its exact output and exit status; the others had none."""
        envelope = {run["name"]: run for run in self.envelope["runs"]}
        paired = [run for run in self.runs if run["f9f78b5"]]
        unpaired = sorted(run["name"] for run in self.runs if not run["f9f78b5"])
        comparison = self.record["comparisons"]["f9f78b5"]
        self.assertEqual(comparison["identical"], len(paired))
        self.assertEqual(comparison["without_counterpart"], unpaired)
        for name in unpaired:
            self.assertNotIn(name, envelope)
        self.require_campaign()
        changes = []
        for run in paired:
            with self.subTest(run=run["name"]):
                old = envelope[run["name"]]
                pair = run["f9f78b5"]
                self.assertEqual(pair["record"], ENVELOPE.name)
                self.assertEqual(pair["log"], f"{self.envelope['campaign_directory']}/{old['log']}")
                self.assertEqual(pair["status"], old["status"])
                self.assertEqual(old["status"], run["status"])
                new_text, old_text = self.text(run["log"]), self.text(pair["log"])
                self.assertTrue(old_text.startswith(f"== source {self.envelope['source_commit']} "))
                self.assertEqual(body(new_text), body(old_text))
                self.assertIs(pair["identical_output"], True)
                self.assertEqual(pair["output_lines"], len(body(new_text)))
                self.assertEqual(exit_status(old_text), run["exit"])
                self.assertEqual(peak_footprint(old_text), pair["peak_memory_footprint_bytes"])
                changes.append((run["peak_memory_footprint_bytes"] / pair["peak_memory_footprint_bytes"] - 1, run["name"], run["role"]))
        summary = comparison["peak_memory_change"]
        for key, chosen in (("all", changes), ("gates", [change for change in changes if change[2] == "gate"])):
            with self.subTest(peaks=key):
                self.assertEqual(summary[key]["runs"], len(chosen))
                self.assertEqual((summary[key]["min"], summary[key]["min_run"]), (round(min(chosen)[0], 4), min(chosen)[1]))
                self.assertEqual((summary[key]["max"], summary[key]["max_run"]), (round(max(chosen)[0], 4), max(chosen)[1]))

    def test_f32_runs_repeat_the_pre_silu_statistics(self):
        """Every F32 run repeats each printed statistic and greedy count of its Mac validation run verbatim."""
        mac = {run["log"]: run for run in self.mac["runs"]}
        roles = {source["commit"]: source["role"] for source in self.record["sources"]}
        f32 = [run for run in self.runs if ".f32." in run["name"]]
        self.assertEqual([run for run in self.runs if "pre_silu" in run], f32)
        self.assertEqual(self.record["comparisons"]["pre_silu_f32"]["runs"], len(f32))
        for run in f32:
            prior = run["pre_silu"]
            self.assertEqual(prior["record"], MAC.name)
            self.assertEqual(mac[prior["log"]]["source_commit"], prior["source_commit"])
            self.assertEqual(mac[prior["log"]]["status"], "pass")
            self.assertEqual(roles[prior["source_commit"]], "comparison (pre-SiLU)")
        self.require_campaign()
        for run in f32:
            with self.subTest(run=run["name"]):
                prior = run["pre_silu"]
                new_lines, old_lines = body(self.text(run["log"])), body(self.text(prior["log"]))
                old_stats, new_stats = stat_lines(old_lines), stat_lines(new_lines)
                self.assertEqual([line for line in old_stats if line not in new_stats], [])
                self.assertEqual(prior["statistic_lines_repeated"], len(old_stats))
                self.assertEqual(prior["new_statistic_lines"], [line for line in new_stats if line not in old_stats])
                self.assertEqual(greedy_counts(old_lines), greedy_counts(new_lines))
                self.assertEqual(prior["greedy"], greedy_counts(new_lines))
                self.assertTrue(old_stats or prior["greedy"], "each comparison judges something")
                self.assertEqual(exit_status(self.text(prior["log"])), run["exit"])

    def test_bundles_and_checkpoints_carry_pinned_identities(self):
        """Each input the runs named has the digest an earlier record pinned and the identity table measured."""
        named, checkpoints = set(), set()
        for run in self.runs:
            for key, value in run["env"].items():
                if key.endswith(("_REFERENCE_DIR", "_DECODE_TRUTH_DIR")):
                    named.add(value)
                elif key.endswith("_OFFICIAL_DIRS"):
                    named.update(value.split(":"))
                elif key in ("TRANSFORMERS_LLAMA_FAMILY_DIR", "TRANSFORMERS_GEMMA_DIR"):
                    checkpoints.add(value)
        bundles = {row["path"]: row for row in self.record["bundles"]}
        self.assertEqual(set(bundles), named)
        mac_bundles = {entry["path"]: entry for entry in self.mac["bundles"]}
        mac_checkpoints = {entry["path"]: entry for entry in self.mac["checkpoints"]}
        rows = {row["path"]: row for row in self.record["checkpoints"]}
        self.assertEqual(set(rows), checkpoints)
        for path, row in rows.items():
            pinned = mac_checkpoints[path]
            self.assertEqual((row["key"], row["repository"], row["revision"], row["config_sha256"]),
                             (pinned["key"], pinned["repository"], pinned["revision"], pinned["config_sha256"]))
        configs = {entry["revision"]: entry["config_sha256"] for entry in rows.values()}
        for row in bundles.values():
            self.assertEqual(configs[row["revision"]], row["config_sha256"], row["path"])
            if row["pinned_by"]["file"] is None:
                self.assertEqual(row["pinned_by"]["record"], MAC.name)
                self.assertEqual(mac_bundles[row["path"]]["manifest_sha256"], row["reference_json_sha256"])
        self.require_campaign()
        identities = json.loads(self.text(self.record["identities"]["file"]))
        self.assertEqual(set(identities["bundles"]), named)
        self.assertEqual(set(identities["checkpoints"]), checkpoints)
        envelope_digests = json.loads(self.text(f"{self.envelope['campaign_directory']}/bundles/bundle-digests.json"))
        round2 = {entry["bundle"]: entry for entry in json.loads(self.text(f"{self.envelope['campaign_directory']}/{self.envelope['bundles']['round2_file_digests']['file']}"))}
        for path, row in bundles.items():
            with self.subTest(bundle=path):
                measured = identities["bundles"][path]
                for key in ("dtype", "device", "attention_implementation", "revision", "config_sha256", "script_sha256", "reference_json_sha256"):
                    self.assertEqual(measured[key], row[key], key)
                pinned = row["pinned_by"]["file"]
                if pinned is not None and pinned.endswith("/bundle-digests.json"):
                    self.assertEqual(envelope_digests[Path(path).name]["reference.json"]["sha256"], row["reference_json_sha256"])
                elif pinned is not None:
                    self.assertEqual(round2[path]["reference_json_sha256"], row["reference_json_sha256"])
        for path, row in rows.items():
            self.assertEqual(identities["checkpoints"][path]["config_sha256"], row["config_sha256"])
        printed = 0
        for run in self.runs:
            for path, digest in re.findall(r"(/Volumes/\S+) (?:\(Float\d+\): )?reference.json SHA-256 ([0-9a-f]{64})", self.text(run["log"])):
                self.assertEqual(bundles[path]["reference_json_sha256"], digest, path)
                printed += 1
        self.assertGreater(printed, 0, "the calibration logs print bundle digests")

    def test_binaries_are_the_envelope_round_binaries(self):
        """The rebuilt binaries equal the envelope round's, the build log names them, and they are still on disk."""
        envelope = {(binary["path"], binary["sha256"]): binary for binary in self.envelope["binaries"]}
        for binary in self.record["binaries"]:
            with self.subTest(binary=binary["target"]):
                old = envelope[(binary["path"], binary["sha256"])]
                self.assertEqual((old["label"], old["target"]), (binary["envelope_record_label"], binary["target"]))
                self.assertEqual(binary["source_commit"], self.head)
                self.assertEqual(binary["built_from"], self.record["checkout"]["path"])
        self.assertEqual({run["binary_sha256"] for run in self.runs}, set(self.binaries))
        self.require_campaign()
        build = self.text(self.record["build"]["rebuild"]["log"])
        self.assertTrue(build.startswith(f"== source {self.head} dirty=0\n"))
        self.assertIn(f"== source after build {self.head} dirty=0", build)
        identities = json.loads(self.text(self.record["identities"]["file"]))["binaries"]
        for binary in self.record["binaries"]:
            with self.subTest(binary=binary["target"]):
                self.assertIn(f"== binary {binary['path']} sha256 {binary['sha256']}\n", build)
                measured = identities[binary["path"]]
                self.assertEqual((measured["sha256_now"], measured["sha256_in_logs"]), (binary["sha256"], [binary["sha256"]]))
                self.assertIs(binary["retained"], True)

    def test_decisions_repeat_the_decision_stage_verbatim(self):
        """The decisions, their rationale, and their review flags are the decision stage's, and review items await the maintainer."""
        decisions = self.record["decisions"]
        self.assertEqual(len({decision["id"] for decision in decisions}), len(decisions))
        awaiting = {entry["decision"] for entry in self.record["awaiting_maintainer"]}
        commits = {source["commit"] for source in self.record["sources"]}
        for decision in decisions:
            with self.subTest(decision=decision["id"]):
                self.assertIsInstance(decision["user_review"], bool)
                self.assertEqual(decision["user_review"], decision["id"] in awaiting)
                self.assertTrue(decision["rationale"] and decision["alternatives_rejected"])
                self.assertLessEqual(set(decision["implementation"]["commits"]), commits)
        self.require_campaign()
        stage = json.loads(self.text(self.record["decisions_source"]["file"]))
        self.assertEqual([{field: decision[field] for field in DECISION_FIELDS} for decision in decisions], stage["decisions"])
        self.assertTrue(stage["implementation_plan"])
        for report in self.record["stage_reports"]:
            self.assertIn(report["path"], self.record["files"])
        for name in ("implement", "rerun"):
            report = json.loads(self.text(f"{self.campaign}/stage-reports/{name}.json"))
            self.assertEqual(report["head"], self.head)

    def test_every_quoted_line_is_in_its_log(self):
        """Each mapping anywhere in the record that pairs a log with quoted lines quotes only lines that log holds."""
        entries = []

        def walk(value, where):
            """Collects every mapping that pairs a log with quoted lines."""
            if isinstance(value, dict):
                if "log" in value and "quotes" in value:
                    entries.append((where, value))
                for key, item in value.items():
                    walk(item, f"{where}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{where}[{index}]")

        walk(self.record, "record")
        self.assertGreater(len(entries), len(self.runs), "the summary quotes its evidence beyond the runs")
        self.require_campaign()
        for where, entry in entries:
            with self.subTest(entry=where):
                self.assertTrue(entry["quotes"])
                log = self.text(entry["log"])
                for quote in entry["quotes"]:
                    self.assertIn(quote, log)

    def test_highlights_are_recorded_runs(self):
        """Every highlighted line belongs to a recorded run, with that run's status, and is in its log."""
        runs = {run["name"]: run for run in self.runs}
        self.assertTrue(self.record["highlights"])
        self.require_campaign()
        for highlight in self.record["highlights"]:
            with self.subTest(highlight=highlight["run"]):
                run = runs[highlight["run"]]
                self.assertEqual(highlight["status"], run["status"])
                self.assertIn(highlight["quote"], self.text(run["log"]))

    def test_final_head_checks_ran_at_the_head(self):
        """The Python tests, the documentation audit, and rustfmt ran clean at the head and exited 0."""
        checks = self.record["final_head_checks"]
        self.assertEqual(checks["source_commit"], self.head)
        self.require_campaign()
        summary = self.text(checks["summary"]["log"])
        self.assertTrue(summary.startswith(f"head {self.head} dirty=0\n"))
        for check in checks["checks"]:
            with self.subTest(check=check["name"]):
                self.assertRegex(summary, rf"(?m)^{re.escape(check['name'])} exit=0$")
                log = self.text(check["log"])
                if "bytes" in check:
                    self.assertEqual(len(log.encode()), check["bytes"])
        audit = self.text(next(check["log"] for check in checks["checks"] if check["name"] == "doc-audit"))
        self.assertTrue(audit.strip())
        self.assertEqual(re.findall(r"(?m)^\w+: (\d+)/\d+ authored sites missing", audit), ["0"] * len(audit.strip().splitlines()))

    def test_open_findings_and_historical_records(self):
        """The carried findings are the envelope record's, and the records this one supersedes keep their bytes."""
        self.assertLessEqual(set(self.record["open_findings_carried"]), {finding["id"] for finding in self.envelope["findings"]})
        self.assertEqual(set(self.record["historical_records"]), HISTORICAL)
        for name, digest in self.record["historical_records"].items():
            with self.subTest(record=name):
                self.assertEqual(sha256((MEASUREMENTS / name).read_bytes()), digest)
        self.assertEqual(self.record["implementation"]["record_sha256"], self.record["historical_records"][ENVELOPE.name])
        self.assertEqual(self.record["implementation"]["tallies"], self.envelope["tallies"])


if __name__ == "__main__":
    unittest.main()
