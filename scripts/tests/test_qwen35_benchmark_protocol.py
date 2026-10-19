"""Tests for the Qwen3.5-0.8B benchmark campaign protocol, on synthetic campaigns.

No binary, checkpoint, or GPU is needed: the plan, host waiting, shader-cache
accounting, steady state, statistics, and judging are pure functions of
their inputs, and the reports here carry only the fields the protocol reads.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "qwen35_benchmark_protocol.py"
SPEC = importlib.util.spec_from_file_location("qwen35_benchmark_protocol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)

#: Two binaries in the order the plan receives them.
BINARIES = [
    {"label": "head", "source": "head source", "binary": "/bin/head-binary", "sha256": "a" * 64},
    {"label": "base", "source": "base source", "binary": "/bin/base-binary", "sha256": "b" * 64},
]
#: Source commit of each label's executable.
COMMITS = {"head": "1" * 40, "base": "2" * 40}
#: The compiler the synthetic executables record: ``rustc -vV`` text and rustup toolchain.
RUSTC = "rustc 1.92.0 (ded5c06cf 2025-12-08)\nrelease: 1.92.0\nLLVM version: 21.1.3"
TOOLCHAIN = "1.92.0-aarch64-apple-darwin"


def distribution(value):
    """A harness distribution of ten repetitions whose median is ``value``."""
    samples = [value * factor for factor in (0.98, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.01, 1.02, 1.5)]
    samples[4] = samples[5] = value
    return {"samples_seconds": samples, "median_seconds": value}


def report(kind, scale=1.0, cold=0.3, tokens="t" * 64, os_build="26A434", label="head", rustc=RUSTC):
    """A minimal benchmark report of ``kind`` by ``label``'s executable whose timings are multiplied by ``scale``."""
    natural = {"total_time": distribution(1.2 * scale), "tokens_u32le_sha256": tokens}
    result = {
        "benchmark": PROTOCOL.REPORT_NAMES[kind],
        "process_memory_instrumentation_enabled": False,
        "timing_comparable_to_uninstrumented_series": True,
        "strict_load_seconds": 1.0 * scale,
        "cold_first_token": {"seconds": cold, "tokens_u32le_sha256": tokens},
        "natural_greedy": natural,
        "protocol": {"measured_repetitions": 10, "warmup_repetitions": 3,
                     "generation": {"strategy": "greedy", "max_new_tokens": 32, "eos_token_ids": [1],
                                    "logit_bias": {}}},
        "provenance": {"cargo_features": ["std", "metal"], "profile": "release", "backend_profile": "Metal/F16",
                       "repository": "Qwen/Qwen3.5-0.8B", "revision": "r" * 40,
                       "benchmark_executable_sha256": next(binary["sha256"] for binary in BINARIES
                                                           if binary["label"] == label),
                       "crate_commit": {"error": None, "value": COMMITS[label]},
                       "crate_worktree_status": {"error": None, "value": ""},
                       "rustc": {"error": None, "value": rustc}, "compiled_rustup_toolchain": TOOLCHAIN,
                       "environment": {"machine_model": "Mac15,11", "operating_system": f"macOS 27.0.1 ({os_build})"}},
        "warm_series": [],
    }
    if kind == "baseline":
        del result["protocol"]["generation"]
        for length in PROTOCOL.SOURCE_LENGTHS:
            series = {"source_length": length, "predicted_tokens_u32le_sha256": tokens,
                      "per_repetition_prediction_hashes": [tokens] * 10}
            for index, metric in enumerate(PROTOCOL.WARM_METRICS):
                series[metric] = distribution((index + 1) * length / 1000 * scale)
            result["warm_series"].append(series)
    return result


def campaign(rounds=8, scale=None, cold=None, growth=None, attempts=None, rustc=RUSTC):
    """Loaded samples of a synthetic campaign.

    ``scale(label, round)`` and ``cold(label, round)`` shape the timings,
    ``growth(label, round)`` the shader-cache growth, and ``attempts(label,
    round)`` lists the shader-cache growth of each discarded attempt run
    before the sample.
    """
    scale = scale or (lambda label, number: 1.0)
    cold = cold or (lambda label, number: 0.3)
    growth = growth or (lambda label, number: 1000)
    attempts = attempts or (lambda label, number: [])
    samples = []
    for sample in PROTOCOL.make_plan(rounds, BINARIES, "/models/qwen", "/models/reference"):
        label, number, kind = sample["label"], sample["round"], sample["benchmark"]
        data = report(kind, scale(label, number), cold(label, number), label=label, rustc=rustc)
        samples.append({"round": number, "position": sample["position"], "label": label, "benchmark": kind,
                        "metrics": PROTOCOL.run_metrics(data, kind), "tokens": PROTOCOL.token_identity(data, kind),
                        "series": PROTOCOL.series_identity(data, kind), "binary": PROTOCOL.binary_identity(data),
                        "metal_cache_growth_bytes": growth(label, number),
                        "discarded_attempts": [{"kind": "contaminated", "metal_cache_growth_bytes": value}
                                               for value in attempts(label, number)],
                        "admission": {}})
    return samples


def series_of(*values):
    """A one-series ``steady_state`` input from per-round values."""
    return {("a", "public", "strict_load"): list(values)}


def summary_of(samples):
    """:func:`summarize` with the campaign's series identity, as ``campaign_summary`` builds it."""
    summary = PROTOCOL.summarize(samples)
    summary["series"] = PROTOCOL.campaign_series(samples)
    return summary


class PlanTests(unittest.TestCase):
    """The plan alternates binaries and counterbalances benchmark order and position."""

    def test_binaries_alternate_inside_every_round(self):
        """Each round runs X Y X Y, both benchmarks of each binary once."""
        plan = PROTOCOL.make_plan(8, BINARIES, "/m", "/r")
        self.assertEqual(len(plan), 32)
        for number in range(1, 9):
            runs = [sample for sample in plan if sample["round"] == number]
            labels = [sample["label"] for sample in runs]
            self.assertNotEqual(labels[0], labels[1])
            self.assertEqual(labels[0], labels[2])
            self.assertEqual(labels[1], labels[3])
            self.assertEqual(sorted((sample["label"], sample["benchmark"]) for sample in runs),
                             sorted((label, kind) for label in ("head", "base") for kind in ("baseline", "public")))

    def test_every_block_is_a_latin_square(self):
        """Every (binary, benchmark) cell runs once in every position of each 4-round block."""
        plan = PROTOCOL.make_plan(16, BINARIES, "/m", "/r")
        for block in range(4):
            rows = [sample for sample in plan if (sample["round"] - 1) // 4 == block]
            self.assertEqual(set(PROTOCOL.position_counts(rows).values()), {1})
            self.assertEqual(len(PROTOCOL.position_counts(rows)), 16)
        counts = PROTOCOL.position_counts(PROTOCOL.make_plan(8, BINARIES, "/m", "/r"))
        self.assertEqual(set(counts.values()), {2})

    def test_each_binary_runs_either_benchmark_first(self):
        """Over eight rounds each binary starts with each benchmark in four rounds, and each binary opens four."""
        plan = PROTOCOL.make_plan(8, BINARIES, "/m", "/r")
        first = {}
        openers = {}
        for number in range(1, 9):
            runs = [sample for sample in plan if sample["round"] == number]
            openers[runs[0]["label"]] = openers.get(runs[0]["label"], 0) + 1
            for label in ("head", "base"):
                kind = next(sample["benchmark"] for sample in runs if sample["label"] == label)
                first[(label, kind)] = first.get((label, kind), 0) + 1
        self.assertEqual(set(first.values()), {4})
        self.assertEqual(openers, {"head": 4, "base": 4})

    def test_plan_paths_are_relative_and_argv_runs_one_test(self):
        """Outputs live below the campaign directory and every process runs exactly one test."""
        sample = PROTOCOL.make_plan(8, BINARIES, "/m", "/r")[0]
        self.assertEqual(sample["argv"], ["/bin/head-binary", "--ignored", "--exact", "--nocapture",
                                          "--test-threads", "1", "qwen::qwen35_metal_baseline_benchmark"])
        for key in ("report", "environment", "log", "host"):
            self.assertFalse(Path(sample[key]).is_absolute())
        self.assertEqual(sample["artifacts"], {"TRANSFORMERS_QWEN35_DIR": "/m",
                                               "TRANSFORMERS_QWEN35_REFERENCE_DIR": "/r"})

    def test_invalid_campaigns_are_refused(self):
        """Fewer than 8, more than 16, or partial blocks of rounds, and anything but two labels, are refused."""
        for rounds in (0, 4, 7, 9, 10, 20):
            with self.assertRaisesRegex(PROTOCOL.ProtocolError, "blocks of 4"):
                PROTOCOL.make_plan(rounds, BINARIES, "/m", "/r")
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "exactly two"):
            PROTOCOL.make_plan(8, BINARIES[:1], "/m", "/r")
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "exactly two"):
            PROTOCOL.make_plan(8, [BINARIES[0], BINARIES[0]], "/m", "/r")
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "start at 1"):
            PROTOCOL.round_order(0)


class HostTests(unittest.TestCase):
    """Cooldown, load admission, load parsing, and shader-cache accounting."""

    def fake(self, loads, start=100.0):
        """A clock, a sleep that advances it, and a load source that pops ``loads``."""
        now = [start]
        slept = []

        def sleep(seconds):
            slept.append(seconds)
            now[0] += seconds
        return (lambda: now[0]), sleep, (lambda: loads.pop(0)), slept

    def test_cooldown_then_load_threshold(self):
        """The sample waits out the cooldown, then polls until both load averages admit it."""
        clock, sleep, load, slept = self.fake([[6.0, 5.0, 5.0], [4.5, 5.0, 5.0], [4.0, 6.0, 5.0]])
        admission = PROTOCOL.wait_for_host(50.0, cooldown=120, threshold=4.0, sustained=6.0, poll=15,
                                           maximum_wait=3600, clock=clock, sleep=sleep, load=load)
        self.assertEqual(slept, [70.0, 15, 15])
        self.assertEqual(admission, {"cooldown_seconds": 120.0, "load_wait_seconds": 30.0,
                                     "admitting_load_1m": 4.0, "admitting_load_5m": 6.0})

    def test_a_lull_in_sustained_load_does_not_admit(self):
        """A low 1-minute load under a high 5-minute load (a pause between builds) keeps the sample waiting."""
        clock, sleep, load, slept = self.fake([[3.7, 13.7, 13.9], [3.0, 6.5, 9.0], [3.0, 5.9, 8.0]])
        admission = PROTOCOL.wait_for_host(0.0, cooldown=0, threshold=4.0, sustained=6.0, poll=15,
                                           clock=clock, sleep=sleep, load=load)
        self.assertEqual(slept, [15, 15])
        self.assertEqual(admission["admitting_load_5m"], 5.9)

    def test_no_cooldown_left_and_low_load_admits_at_once(self):
        """A sample after a long gap under low load starts without sleeping."""
        clock, sleep, load, slept = self.fake([[1.0, 1.0, 1.0]], start=1000.0)
        admission = PROTOCOL.wait_for_host(0.0, cooldown=120, threshold=4.0, clock=clock, sleep=sleep, load=load)
        self.assertEqual(slept, [])
        self.assertEqual(admission["load_wait_seconds"], 0.0)

    def test_persistent_load_stops_the_campaign(self):
        """A load that never falls stops the campaign after the maximum wait."""
        clock, sleep, load, _ = self.fake([[9.0, 9.0, 9.0]] * 10)
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "stayed above 4.0"):
            PROTOCOL.wait_for_host(100.0, cooldown=0, threshold=4.0, poll=15, maximum_wait=60,
                                   clock=clock, sleep=sleep, load=load)

    def test_in_run_load_above_the_limit_contaminates(self):
        """Any in-run 1-minute load above the limit marks the attempt contaminated; the limit itself does not."""
        self.assertEqual(PROTOCOL.IN_RUN_LOAD_LIMIT, PROTOCOL.LOAD_THRESHOLD + 2.0)
        self.assertFalse(PROTOCOL.contaminated([[5.0, 3.1], [10.0, 6.0]]))
        self.assertTrue(PROTOCOL.contaminated([[5.0, 3.1], [10.0, 13.6]]))
        self.assertFalse(PROTOCOL.contaminated([]))

    def test_load_averages_parse_sysctl_output(self):
        """``{ a b c }`` parses to three floats; anything else is refused."""
        self.assertEqual(PROTOCOL.load_averages("{ 2.35 6.46 7.52 }"), [2.35, 6.46, 7.52])
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "unreadable"):
            PROTOCOL.load_averages("load: high")

    def test_cache_growth_counts_new_grown_and_removed_files(self):
        """Growth sums size changes of changed files, new files counting whole and removed ones negative."""
        before = {"a.data": [100, 1], "b.data": [50, 1], "gone.list": [10, 1]}
        after = {"a.data": [160, 2], "b.data": [50, 1], "new.list": [5, 2]}
        growth = PROTOCOL.cache_growth(before, after)
        self.assertEqual(growth["growth_bytes"], 60 + 5 - 10)
        self.assertEqual(set(growth["files"]), {"a.data", "gone.list", "new.list"})
        self.assertEqual(PROTOCOL.cache_growth(before, before), {"files": {}, "growth_bytes": 0})

    def test_metal_cache_state_reads_sizes(self):
        """Every file below the cache directory is listed with its size; a missing directory is empty."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sub").mkdir()
            (root / "sub" / "functions.data").write_bytes(b"x" * 7)
            state = PROTOCOL.metal_cache_state(root)
            self.assertEqual(list(state), ["sub/functions.data"])
            self.assertEqual(state["sub/functions.data"][0], 7)
            self.assertEqual(PROTOCOL.metal_cache_state(root / "missing"), {})


class MetricTests(unittest.TestCase):
    """Per-run values come from the harness's raw repetitions; series identity ignores schema growth."""

    def test_baseline_report_yields_every_gated_metric(self):
        """Strict load, twelve warm medians, and the natural loop are gated; cold is the diagnostic."""
        metrics = PROTOCOL.run_metrics(report("baseline"), "baseline")
        self.assertEqual(list(metrics["gated"]), PROTOCOL.gated_metric_names("baseline"))
        self.assertEqual(len(metrics["gated"]), 14)
        self.assertEqual(metrics["diagnostic"], {"cold_first_token": 0.3})
        self.assertEqual(PROTOCOL.gated_metric_names("public"), ["strict_load", "natural_loop"])

    def test_inconsistent_or_instrumented_reports_are_refused(self):
        """A stored median that disagrees with its samples, a wrong count, a wrong kind, or instrumentation fail."""
        data = report("baseline")
        data["warm_series"][0]["prefill"]["median_seconds"] *= 1.01
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "disagrees"):
            PROTOCOL.run_metrics(data, "baseline")
        data = report("public")
        data["natural_greedy"]["total_time"]["samples_seconds"].pop()
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "not 10"):
            PROTOCOL.run_metrics(data, "public")
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "expected a"):
            PROTOCOL.run_metrics(report("public"), "baseline")
        data = report("public")
        data["process_memory_instrumentation_enabled"] = True
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "instrumented"):
            PROTOCOL.run_metrics(data, "public")
        data = report("baseline")
        data["warm_series"].pop()
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "lengths"):
            PROTOCOL.run_metrics(data, "baseline")

    def test_token_identity_requires_identical_repetitions(self):
        """Warm predictions must agree across repetitions."""
        data = report("baseline")
        data["warm_series"][1]["per_repetition_prediction_hashes"][3] = "x" * 64
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "differ between repetitions"):
            PROTOCOL.token_identity(data, "baseline")

    def test_generation_schema_growth_is_not_a_new_series(self):
        """Neutral generation fields do not change the series; the length bound or OS build does."""
        first, second = report("public"), report("public")
        second["protocol"]["generation"]["min_new_tokens"] = 0
        self.assertEqual(PROTOCOL.series_identity(first, "public"), PROTOCOL.series_identity(second, "public"))
        second["protocol"]["generation"]["max_new_tokens"] = 64
        self.assertNotEqual(PROTOCOL.series_identity(first, "public"), PROTOCOL.series_identity(second, "public"))
        self.assertNotEqual(PROTOCOL.series_identity(first, "public"),
                            PROTOCOL.series_identity(report("public", os_build="26B1"), "public"))

    def test_the_compiler_is_part_of_the_series(self):
        """Another rustc or rustup toolchain is another series, and a report without them has none."""
        first = PROTOCOL.series_identity(report("baseline"), "baseline")
        self.assertEqual((first["rustc"], first["rustup_toolchain"]), (RUSTC, TOOLCHAIN))
        other = report("baseline", rustc="rustc 1.99.1 (aaaaaaaaa 2026-09-01)")
        self.assertNotEqual(first, PROTOCOL.series_identity(other, "baseline"))
        other = report("baseline")
        other["provenance"]["compiled_rustup_toolchain"] = "nightly-aarch64-apple-darwin"
        self.assertNotEqual(first, PROTOCOL.series_identity(other, "baseline"))
        for field, value in (("rustc", {"error": "rustc not found", "value": None}),
                             ("compiled_rustup_toolchain", None)):
            missing = report("baseline")
            missing["provenance"][field] = value
            with self.assertRaisesRegex(PROTOCOL.ProtocolError, "rustc version and rustup toolchain"):
                PROTOCOL.series_identity(missing, "baseline")
            with self.assertRaisesRegex(PROTOCOL.ProtocolError, "rustc version and rustup toolchain"):
                PROTOCOL.binary_identity(missing)

    def test_binary_identity_names_the_executable_and_its_build(self):
        """Digest, commit, clean status, compiler, sorted features, and profile identify the executable."""
        identity = PROTOCOL.binary_identity(report("public", label="base"))
        self.assertEqual(identity, {"sha256": "b" * 64, "source_commit": COMMITS["base"], "worktree_status": "",
                                    "rustc": RUSTC, "rustup_toolchain": TOOLCHAIN,
                                    "cargo_features": ["metal", "std"], "profile": "release"})
        missing = report("public")
        missing["provenance"]["crate_commit"] = {"error": "not a git checkout", "value": None}
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "source commit"):
            PROTOCOL.binary_identity(missing)


class SteadyStateTests(unittest.TestCase):
    """Steady state is the longest suffix of whole blocks in which no block drifts beyond tolerance and noise."""

    def test_flat_campaign_is_steady_from_round_one(self):
        """Identical rounds are steady from the first, and nothing limits the window."""
        steady = PROTOCOL.summarize(campaign())["steady_state"]
        self.assertEqual(steady["first_round"], 1)
        self.assertEqual(steady["rounds"], list(range(1, 9)))
        self.assertEqual(steady["maximum_absolute_drift"], 0.0)
        self.assertIsNone(steady["limiting_block"])
        self.assertEqual(steady["drifting"], [])
        self.assertEqual([assessment["first_round"] for assessment in steady["assessments"]], [1])

    def test_a_slow_first_block_is_excluded(self):
        """A 20% slow start excludes its block; one slow round left inside the next block is absorbed."""
        samples = campaign(rounds=12, scale=lambda label, number: 1.2 if number <= 3 else 1.0)
        summary = PROTOCOL.summarize(samples)
        steady = summary["steady_state"]
        self.assertEqual(steady["rounds"], list(range(5, 13)))
        self.assertEqual(steady["limiting_block"], 1)
        self.assertEqual(len(steady["drifting"]), 2 * (14 + 2))
        samples = campaign(rounds=12, scale=lambda label, number: 1.2 if number <= 5 else 1.0)
        summary = PROTOCOL.summarize(samples)
        self.assertEqual(summary["steady_state"]["first_round"], 5)
        self.assertEqual(summary["cells"]["head/public"]["gated"]["strict_load"]["median"], 1.0)
        self.assertEqual(summary["cells"]["head/public"]["gated"]["strict_load"]["maximum"], 1.2)

    def test_a_drifting_series_never_converges(self):
        """A metric that slows 2% per round has no steady window; statistics are then withheld."""
        samples = campaign(scale=lambda label, number: 1.0 + 0.02 * number if label == "base" else 1.0)
        summary = PROTOCOL.summarize(samples)
        steady = summary["steady_state"]
        self.assertIsNone(steady["first_round"])
        self.assertEqual(steady["limiting_block"], 1)
        self.assertTrue(all(name.startswith("base/") for name in steady["drifting"]))
        self.assertEqual(summary["cells"]["base/baseline"]["gated"], {})
        self.assertNotIn("steady_state", summary["cells"]["base/baseline"]["diagnostic"]["cold_first_token"])
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "steady state"):
            PROTOCOL.reference_block(summary, "head")

    def test_a_later_drift_voids_an_earlier_window_that_passed(self):
        """A step in the last block hides in a long window's median, but the later block's drift ends the search.

        From round 1 the first block differs from the later rounds' median by
        only 1.5%, so that window passes on its own; the block from round 5
        then meets the 3% step alone. A rule that took the earliest passing
        window would call the campaign steady from round 1; this one finds no
        steady state, and with one more block at the new level it starts
        after the step.
        """
        steady = PROTOCOL.steady_state(series_of(*[1.0] * 8, *[1.03] * 4), 12)
        self.assertEqual(steady["assessments"][0]["drifting"], [])
        self.assertAlmostEqual(steady["assessments"][0]["series"]["a/public/strict_load"]["drift"], 0.015)
        self.assertEqual(steady["assessments"][1]["drifting"], ["a/public/strict_load"])
        self.assertIsNone(steady["first_round"])
        self.assertEqual(steady["limiting_block"], 5)
        self.assertEqual(steady["rounds"], [])
        steady = PROTOCOL.steady_state(series_of(*[1.0] * 8, *[1.03] * 8), 16)
        self.assertEqual([assessment["drifting"] != [] for assessment in steady["assessments"]], [True, True, False])
        self.assertEqual((steady["first_round"], steady["limiting_block"]), (9, 5))

    def test_drift_tolerance_is_inclusive(self):
        """Without noise, a later median exactly 2.5% above the block is steady and a 3% shift is not."""
        self.assertEqual(PROTOCOL.steady_state(series_of(*[1.0] * 4, *[1.025] * 4), 8)["first_round"], 1)
        steady = PROTOCOL.steady_state(series_of(*[1.0] * 4, *[1.03] * 4), 8)
        self.assertIsNone(steady["first_round"])
        self.assertEqual(steady["drifting"], ["a/public/strict_load"])
        self.assertIsNone(steady["assessments"][0]["series"]["a/public/strict_load"]["z"])
        self.assertEqual(steady["rounds"], [])

    def test_noise_of_single_observations_is_not_drift(self):
        """A 3.4% block difference within one-observation noise is steady; the same shift without noise is not.

        The values are the head's strict loads of the October 18 campaign's
        rounds 5 to 12, whose later block has outliers of 1.02 and 0.97 s.
        """
        loads = [1.0267, 0.9296, 0.9276, 0.9303, 1.0232, 0.9747, 0.9228, 0.949]
        steady = PROTOCOL.steady_state(series_of(*loads), 8)
        item = steady["assessments"][0]["series"]["a/public/strict_load"]
        self.assertGreater(item["drift"], PROTOCOL.STEADY_DRIFT)
        self.assertLess(item["z"], 1.0)
        self.assertEqual(steady["first_round"], 1)
        self.assertEqual((steady["beyond_tolerance"], steady["maximum_z_beyond_tolerance"]), (1, item["z"]))
        self.assertIsNone(PROTOCOL.steady_state(series_of(*[1.0] * 4, *[1.034] * 4), 8)["first_round"])
        shifted = PROTOCOL.steady_state(series_of(1.10, 1.12, 1.09, 1.11, 1.0, 0.99, 1.01, 1.0), 8)
        self.assertIsNone(shifted["first_round"])
        self.assertGreater(shifted["assessments"][0]["series"]["a/public/strict_load"]["z"], 10)

    def test_neither_an_outlier_nor_a_trend_widens_the_limit(self):
        """The noise is the smaller of the outlier-proof and the trend-proof robust scale."""
        outlier = PROTOCOL.robust_noise([1.03, 1.031, 1.29, 1.032])
        self.assertLess(outlier["mad"], 0.002)
        self.assertGreater(outlier["successive"], 0.2)
        self.assertEqual(outlier["noise"], outlier["mad"])
        self.assertTrue(PROTOCOL.block_drift([1.0] * 4, [1.03, 1.031, 1.29, 1.032])["drifting"])
        trend = PROTOCOL.robust_noise([1.02, 1.03, 1.04, 1.05])
        self.assertAlmostEqual(trend["successive"], 0.01 / (PROTOCOL.NORMAL_QUARTILE * 2 ** 0.5) / 1.035)
        self.assertGreater(trend["mad"], trend["successive"])
        self.assertEqual(trend["noise"], trend["successive"])
        drift = PROTOCOL.block_drift([1.0] * 4, [1.02, 1.03, 1.04, 1.05])
        self.assertTrue(drift["drifting"])
        self.assertLess(drift["z"] * trend["successive"] / trend["mad"], PROTOCOL.STEADY_Z)
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "at least two"):
            PROTOCOL.robust_noise([1.0])
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "nonpositive"):
            PROTOCOL.robust_noise([0.0, 0.0])
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "later rounds"):
            PROTOCOL.block_drift([1.0], [1.0])
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "nonpositive"):
            PROTOCOL.block_drift([0.0], [1.0, 1.0])

    def test_seeded_noise_is_steady_and_seeded_shifts_are_found(self):
        """Over seeded random campaigns, stationary noise rarely blocks steady state and a 5% shift never passes."""
        generator = random.Random(20261019)
        unsteady = 0
        for _ in range(200):
            values = [1.0 + generator.gauss(0.0, 0.01) for _ in range(12)]
            unsteady += PROTOCOL.steady_state(series_of(*values), 12)["first_round"] != 1
        self.assertLessEqual(unsteady, 4)
        for _ in range(200):
            values = [(1.05 if number <= 4 else 1.0) + generator.gauss(0.0, 0.003) for number in range(1, 13)]
            self.assertEqual(PROTOCOL.steady_state(series_of(*values), 12)["first_round"], 5)
            values = [(1.05 if number > 8 else 1.0) + generator.gauss(0.0, 0.003) for number in range(1, 13)]
            self.assertIsNone(PROTOCOL.steady_state(series_of(*values), 12)["first_round"])

    def test_windows_are_whole_balanced_blocks(self):
        """Every cell runs equally often in every position of the window; partial blocks are refused."""
        summary = PROTOCOL.summarize(campaign(rounds=12, scale=lambda label, number: 1.2 if number <= 2 else 1.0))
        self.assertEqual(summary["steady_state"]["first_round"], 5)
        counts = summary["steady_state"]["position_counts"]
        self.assertEqual(set(counts), {f"{label}/{kind}" for label in ("head", "base") for kind in PROTOCOL.BENCHMARKS})
        for positions in counts.values():
            self.assertEqual(positions, {"1": 2, "2": 2, "3": 2, "4": 2})
        for rounds in (4, 6, 10):
            with self.assertRaisesRegex(PROTOCOL.ProtocolError, "whole blocks"):
                PROTOCOL.steady_state(series_of(*[1.0] * rounds), rounds)
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "rounds, not"):
            PROTOCOL.steady_state({("a", "b", "c"): [1.0] * 7}, 8)

    def test_cold_first_token_never_decides_steady_state(self):
        """A cold time that halves over the campaign leaves the gated window untouched and is reported."""
        samples = campaign(cold=lambda label, number: 0.8 / number, growth=lambda label, number: 2000 // number)
        summary = PROTOCOL.summarize(samples)
        self.assertEqual(summary["steady_state"]["first_round"], 1)
        cold = summary["cells"]["head/baseline"]["diagnostic"]["cold_first_token"]
        self.assertFalse(cold["gated"])
        self.assertAlmostEqual(cold["all_rounds"]["maximum"], 0.8)
        self.assertAlmostEqual(cold["all_rounds"]["minimum"], 0.1)
        self.assertAlmostEqual(cold["spearman_with_round"], -1.0)
        self.assertAlmostEqual(cold["spearman_with_metal_cache_growth"], 1.0)
        self.assertEqual(cold["rounds_after_discarded_attempts"], [])
        self.assertEqual(cold["without_rounds_after_discarded_attempts"]["all_rounds"], cold["all_rounds"])

    def test_samples_after_discarded_attempts_are_reported_apart(self):
        """A rerun that starts with a cache its discarded attempt warmed is listed and left out of the second view."""
        samples = campaign(cold=lambda label, number: 0.1 if (label, number) == ("head", 3) else 0.3,
                           growth=lambda label, number: 100 if (label, number) == ("head", 3) else 1000,
                           attempts=lambda label, number: [900, 50] if (label, number) == ("head", 3) else [])
        cold = PROTOCOL.summarize(samples)["cells"]["head/public"]["diagnostic"]["cold_first_token"]
        self.assertEqual(cold["rounds_after_discarded_attempts"], [3])
        self.assertEqual(cold["discarded_attempts_metal_cache_growth_bytes"], [0, 0, 950, 0, 0, 0, 0, 0])
        self.assertEqual(cold["all_rounds"]["minimum"], 0.1)
        self.assertEqual(cold["steady_state"]["minimum"], 0.1)
        without = cold["without_rounds_after_discarded_attempts"]
        self.assertEqual(without["all_rounds"]["values"], [0.3] * 7)
        self.assertEqual(without["steady_state"]["values"], [0.3] * 7)
        self.assertIsNone(without["spearman_with_round"])
        self.assertAlmostEqual(cold["spearman_with_metal_cache_growth"], 1.0)
        self.assertIsNotNone(cold["spearman_with_metal_cache_growth_including_discarded_attempts"])

    def test_inconsistent_campaigns_are_refused(self):
        """A missing round, changed tokens inside one cell, or a label with two executables are refused."""
        samples = campaign()
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "one run per round"):
            PROTOCOL.summarize(samples[1:])
        changed = copy.deepcopy(samples)
        changed[5]["tokens"] = {"cold": "different"}
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "token identities differ"):
            PROTOCOL.summarize(changed)
        changed = copy.deepcopy(samples)
        changed[6]["binary"]["sha256"] = "c" * 64
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "more than one executable"):
            PROTOCOL.summarize(changed)

    def test_rank_statistics(self):
        """Ties share ranks; constant or short inputs have no correlation; no values have no statistics."""
        self.assertEqual(PROTOCOL.ranks([3.0, 1.0, 3.0, 2.0]), [3.5, 1.0, 3.5, 2.0])
        self.assertIsNone(PROTOCOL.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        self.assertIsNone(PROTOCOL.spearman([1.0, 2.0], [1.0, 2.0]))
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "paired"):
            PROTOCOL.spearman([1.0], [1.0, 2.0])
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "no values"):
            PROTOCOL.median([])
        self.assertIsNone(PROTOCOL.statistics_of([]))


class JudgeTests(unittest.TestCase):
    """Gated metrics alert above 5% and fail above 10% only after a controlled rerun; cold never gates."""

    @classmethod
    def setUpClass(cls):
        """A flat reference campaign for the label ``head``."""
        cls.reference = PROTOCOL.reference_block(summary_of(campaign()), "head")

    def verdict(self, scale, cold=0.3, reference_binary=None, base_scale=1.0):
        """Judge a campaign whose ``base`` label is the candidate (``head`` reruns the reference)."""
        samples = campaign(scale=lambda label, number: scale if label == "base" else base_scale,
                           cold=lambda label, number: cold if label == "base" else 0.3)
        return PROTOCOL.judge(self.reference, summary_of(samples), "base", reference_binary)

    def test_reference_block_holds_what_judging_reads(self):
        """The binary, every gated metric, the tokens, the series, and the ungated cold statistics are recorded."""
        block = self.reference
        self.assertEqual(block["label"], "head")
        self.assertEqual(block["binary"], PROTOCOL.binary_identity(report("baseline", label="head")))
        self.assertEqual(block["series"]["baseline"]["rustc"], RUSTC)
        self.assertEqual(set(block["benchmarks"]["baseline"]["gated"]), set(PROTOCOL.gated_metric_names("baseline")))
        cold = block["benchmarks"]["public"]["diagnostic"]["cold_first_token"]
        self.assertFalse(cold["gated"])
        self.assertEqual(cold["median"], 0.3)
        self.assertEqual(set(block["series"]), {"baseline", "public"})

    def test_unchanged_candidate_is_ok_and_cold_is_only_reported(self):
        """Equal timings pass; a tripled cold first token is reported with its median and range, never gated."""
        result = self.verdict(1.0, cold=0.9)
        self.assertEqual(result["overall"], "ok")
        cold = result["metrics"]["baseline/cold_first_token"]
        self.assertEqual(cold["status"], "diagnostic")
        self.assertFalse(cold["gated"])
        self.assertEqual((cold["median"], cold["range"], cold["reference_median"]), (0.9, [0.9, 0.9], 0.3))

    def test_changes_above_five_percent_alert_either_way(self):
        """6% slower or 6% faster alerts; 4% does not."""
        self.assertEqual(self.verdict(1.06)["overall"], "alert")
        self.assertEqual(self.verdict(0.94)["overall"], "alert")
        self.assertEqual(self.verdict(1.04)["overall"], "ok")
        self.assertEqual(PROTOCOL.classify(1.05), "ok")
        self.assertEqual(PROTOCOL.classify(1.10), "alert")

    def test_slowdown_above_ten_percent_is_only_a_candidate_without_a_rerun(self):
        """12% slower is a failure candidate until a controlled rerun."""
        result = self.verdict(1.12)
        self.assertEqual(result["overall"], "failure_candidate")
        self.assertEqual(result["metrics"]["public/strict_load"]["status"], "failure_candidate")
        self.assertAlmostEqual(result["metrics"]["public/strict_load"]["ratio"], 1.12)

    def test_controlled_rerun_confirms_a_failure(self):
        """With the reference binary steady at its record, a 12% slower candidate fails."""
        result = self.verdict(1.12, reference_binary="head")
        self.assertTrue(result["controlled_rerun"])
        self.assertEqual(result["reference_binary"], {"label": "head", "matched_by": "sha256"})
        self.assertEqual(result["overall"], "fail")
        self.assertEqual(result["metrics"]["baseline/natural_loop"]["status"], "fail")

    def test_a_clean_rebuild_of_the_reference_source_may_rerun_it(self):
        """Another executable from the same commit, clean tree, compiler, features, and profile is the reference binary."""
        reference = copy.deepcopy(self.reference)
        reference["binary"]["sha256"] = "f" * 64
        samples = campaign(scale=lambda label, number: 1.12 if label == "base" else 1.0)
        result = PROTOCOL.judge(reference, summary_of(samples), "base", "head")
        self.assertEqual(result["reference_binary"], {"label": "head", "matched_by": "source_and_toolchain"})
        self.assertEqual(result["overall"], "fail")

    def test_a_rerun_of_another_binary_cannot_confirm_or_clear_a_failure(self):
        """A rerun label that is neither the recorded executable nor a clean rebuild of its source is refused.

        Pairing a 12% slower candidate with any other binary that is 12% slow
        too would read as series drift and hide the regression, so each
        mismatch is refused before any verdict.
        """
        summary = summary_of(campaign(scale=lambda label, number: 1.12))
        for field, value in (("source_commit", "9" * 40), ("rustc", "rustc 1.99.1"),
                             ("rustup_toolchain", "1.99.1-aarch64-apple-darwin"), ("cargo_features", ["std"]),
                             ("profile", "debug"), ("worktree_status", " M src/lib.rs")):
            with self.subTest(field=field):
                reference = copy.deepcopy(self.reference)
                reference["binary"]["sha256"] = "f" * 64
                reference["binary"][field] = value
                with self.assertRaisesRegex(PROTOCOL.ProtocolError, "is not the reference binary"):
                    PROTOCOL.judge(reference, summary, "base", "head")
        dirty = copy.deepcopy(summary)
        dirty["binaries"]["head"]["sha256"] = "f" * 64
        dirty["binaries"]["head"]["worktree_status"] = " M src/lib.rs"
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "is not the reference binary"):
            PROTOCOL.judge(self.reference, dirty, "base", "head")
        retokened = copy.deepcopy(summary)
        retokened["cells"]["head/public"]["tokens"] = {"cold": "other", "natural": "other"}
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "does not reproduce the reference's public tokens"):
            PROTOCOL.judge(self.reference, retokened, "base", "head")
        unrecorded = copy.deepcopy(self.reference)
        del unrecorded["binary"]
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "records no binary identity"):
            PROTOCOL.judge(unrecorded, summary, "base", "head")
        self.assertEqual(PROTOCOL.judge(unrecorded, summary, "base")["overall"], "failure_candidate")
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "its own reference binary"):
            PROTOCOL.judge(self.reference, summary, "base", "base")
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "no label 'other'"):
            PROTOCOL.judge(self.reference, summary, "base", "other")
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "no candidate label 'other'"):
            PROTOCOL.judge(self.reference, summary, "other")

    def test_host_drift_is_not_a_code_regression(self):
        """When the reference binary is 12% slower too, the series drifted: re-baselining is a maintainer decision."""
        result = self.verdict(1.13, reference_binary="head", base_scale=1.12)
        self.assertEqual(result["overall"], "series_drift")
        result = self.verdict(1.12, reference_binary="head", base_scale=1.05)
        self.assertEqual(result["overall"], "not_confirmed")

    def test_incomparable_candidates_are_refused(self):
        """A different series, different tokens, or an unsteady campaign cannot be judged."""
        reference = copy.deepcopy(self.reference)
        reference["series"]["public"]["operating_system"] = "macOS 15.7.9 (24G1)"
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "different benchmark series"):
            PROTOCOL.judge(reference, summary_of(campaign()), "base")
        reference = copy.deepcopy(self.reference)
        reference["benchmarks"]["baseline"]["tokens"]["cold"] = "other"
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "tokens differ"):
            PROTOCOL.judge(reference, summary_of(campaign()), "base")
        unsteady = summary_of(campaign(scale=lambda label, number: 1.0 + 0.02 * number))
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "did not reach steady state"):
            PROTOCOL.judge(self.reference, unsteady, "base")
        recompiled = summary_of(campaign(rustc="rustc 1.99.1 (aaaaaaaaa 2026-09-01)"))
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "different benchmark series"):
            PROTOCOL.judge(self.reference, recompiled, "base")
        with self.assertRaisesRegex(PROTOCOL.ProtocolError, "different benchmark series"):
            PROTOCOL.judge(self.reference, recompiled, "base", "head")


class CampaignFileTests(unittest.TestCase):
    """A campaign directory round-trips through ``load_campaign`` and the command line."""

    def test_load_and_summarize_a_campaign_directory(self):
        """Reports and host records written to disk give the same summary as the in-memory campaign."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = PROTOCOL.make_plan(8, BINARIES, "/m", "/r")
            for sample in samples:
                data = report(sample["benchmark"], label=sample["label"])
                for key in ("report", "host"):
                    (root / sample[key]).parent.mkdir(parents=True, exist_ok=True)
                (root / sample["report"]).write_text(json.dumps(data))
                (root / sample["host"]).write_text(json.dumps({
                    "exit": 0, "binary_sha256_before": sample["sha256"],
                    "metal_cache_growth": {"growth_bytes": 1000}, "admission": {}}))
            (root / "plan.json").write_text(json.dumps({"samples": samples}))
            stem = Path(samples[3]["host"]).name.removesuffix(".host.json")
            attempt = root / "host" / f"{stem}.contaminated-2026-10-19T000000+0000.host.json"
            attempt.write_text(json.dumps({"sample": samples[3], "host_before": {"utc": "2026-10-19T00:00:00+00:00"},
                                           "metal_cache_growth": {"growth_bytes": 500}, "clean": False}))
            (root / "host" / f"{stem}.summary.host.json").write_text("{}")
            loaded = PROTOCOL.load_campaign(root)
            self.assertEqual(loaded[3]["discarded_attempts"], [{
                "host": f"host/{attempt.name}", "kind": "contaminated", "started_utc": "2026-10-19T00:00:00+00:00",
                "metal_cache_growth_bytes": 500}])
            self.assertEqual(sum(len(sample["discarded_attempts"]) for sample in loaded), 1)
            self.assertEqual(loaded[3]["binary"], PROTOCOL.binary_identity(report("public", label=samples[3]["label"])))
            summary = PROTOCOL.campaign_summary(root)
            self.assertEqual(summary["steady_state"]["first_round"], 1)
            self.assertEqual(summary["series"], PROTOCOL.campaign_series(campaign()))
            self.assertEqual(set(summary["binaries"]), {"head", "base"})
            stray = json.loads(attempt.read_text())
            stray["sample"] = samples[4]
            attempt.write_text(json.dumps(stray))
            with self.assertRaisesRegex(PROTOCOL.ProtocolError, "records another sample"):
                PROTOCOL.load_campaign(root)
            attempt.unlink()
            self.assertEqual(PROTOCOL.main(["summarize", str(root)]), 0)
            self.assertEqual(json.loads((root / "summary.json").read_text())["rounds"], 8)
            host = json.loads((root / samples[3]["host"]).read_text())
            host["exit"] = 101
            (root / samples[3]["host"]).write_text(json.dumps(host))
            with self.assertRaisesRegex(PROTOCOL.ProtocolError, "failed or unauthenticated"):
                PROTOCOL.load_campaign(root)
            host["exit"] = 0
            host["clean"] = False
            (root / samples[3]["host"]).write_text(json.dumps(host))
            with self.assertRaisesRegex(PROTOCOL.ProtocolError, "contaminated"):
                PROTOCOL.load_campaign(root)
            host["clean"] = True
            (root / samples[3]["host"]).write_text(json.dumps(host))
            data = json.loads((root / samples[3]["report"]).read_text())
            data["provenance"]["benchmark_executable_sha256"] = "c" * 64
            (root / samples[3]["report"]).write_text(json.dumps(data))
            with self.assertRaisesRegex(PROTOCOL.ProtocolError, "another executable"):
                PROTOCOL.load_campaign(root)
            data["provenance"]["benchmark_executable_sha256"] = samples[3]["sha256"]
            data["provenance"]["environment"]["operating_system"] = "macOS 27.1 (26B1)"
            (root / samples[3]["report"]).write_text(json.dumps(data))
            with self.assertRaisesRegex(PROTOCOL.ProtocolError, "mixes"):
                PROTOCOL.load_campaign(root)
            data["provenance"]["environment"]["operating_system"] = "macOS 27.0.1 (26A434)"
            data["provenance"]["rustc"]["value"] = "rustc 1.99.1 (aaaaaaaaa 2026-09-01)"
            (root / samples[3]["report"]).write_text(json.dumps(data))
            with self.assertRaisesRegex(PROTOCOL.ProtocolError, "mixes .* benchmark series .*compiler"):
                PROTOCOL.load_campaign(root)


if __name__ == "__main__":
    unittest.main()
