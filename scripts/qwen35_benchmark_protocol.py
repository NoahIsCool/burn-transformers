#!/usr/bin/env python3
"""Plan, run, summarize, and judge Qwen3.5-0.8B Metal/F16 benchmark campaigns.

This is the executable form of the campaign protocol in ``BENCHMARKS.md``
("Qwen3.5-0.8B benchmark campaign protocol"). A campaign compares exactly two
prebuilt ``qwen35_benchmarks`` release binaries on one host. Each round runs
both binaries' two benchmarks once (four processes), and every process runs
one test:

* ``baseline``: ``qwen::qwen35_metal_baseline_benchmark`` (32/128/512-token
  prefill, 128 forced cached decode steps, and the natural model-API loop);
* ``public``: ``qwen::qwen35_metal_public_causal_greedy_benchmark`` (the
  public ``generate_causal`` coordinator).

Design. The two binaries alternate inside every round (X Y X Y), and the
eight round layouts below form two 4 x 4 Latin squares over the four
(binary, benchmark) cells: every cell runs once in every position of each
4-round block, so neither binary nor benchmark is confounded with its
position in the round, and inside a round each binary always follows the
other.
Campaigns take whole blocks: at least 8 rounds and at most 16.

Host control. Before every sample the runner waits at least the cooldown
since the previous process ended (and since the campaign started), then until
the 1-minute load average is at or below the load threshold and the 5-minute
average at or below the sustained-load threshold, so a short lull in other
work does not admit a sample. While the process runs it samples the 1-minute
load average every few seconds; a sample during which it rose above the
in-run limit (the admission threshold plus the benchmark's own two runnable
threads) is contaminated by other work. Its files are kept under
``contaminated`` names and the sample is run again, at most four attempts in
all. The runner requires AC power, both external volumes, an unchanged host
and OS identity, and the planned binary digest, and records load, power,
thermal state, the top CPU processes, and the size of every file in the
per-user Metal shader cache before and after the process. The runner never
takes the heavy-model lock; its caller does.

Statistics. A run's value of a warm or natural-loop metric is the median of
its ten measured repetitions; strict load and cold first token are one
observation per process. Steady state is assessed over whole blocks, working
back from the end of the campaign: a block drifts when, for some cell and
gated metric, the median of its four rounds differs from the median of all
later rounds by more than 2.5% (half the alert threshold) and by more than
three standard errors of that difference, estimated from the later rounds'
robust round-to-round noise. The window starts at the earliest block from
which no block drifts and spans at least two blocks, so every later block
has passed too, and every cell runs equally often in every position of the
window. When the second-to-last block already drifts against the last, the
campaign has no steady state and needs more rounds. Gated statistics are
medians over the steady-state rounds. Cold first token is reported as the
median and range over the same rounds and over all rounds, with its rank
correlations with shader-cache growth and with the round number, also
without the samples that followed a discarded (contaminated or failed)
attempt, which grew the shader cache before them; it is a diagnostic and
never alerts or fails.

Judging. Against a recorded reference, a gated metric whose candidate median
changes by more than 5% in either direction alerts; one that is more than
10% slower is a failure candidate. A failure is confirmed only by a
controlled rerun: a campaign of this protocol that alternates the candidate
with the reference binary (the recorded executable, by SHA-256, or a clean
build of its source commit with the same compiler, features, and profile,
reproducing the reference tokens) and finds the candidate more than 10%
slower than both the recorded reference value and the reference binary's
median in that campaign. When the reference binary is itself more than 10%
slower than its recorded value, the series has drifted (host change), and
re-baselining is a maintainer decision. A candidate built with another
compiler is another series and is not judged.

Usage::

    qwen35_benchmark_protocol.py plan CAMPAIGN --rounds 8 --model-dir DIR \\
        --reference-dir DIR --accelerator GPU LABEL=SOURCE=BINARY LABEL=SOURCE=BINARY
    qwen35_benchmark_protocol.py run CAMPAIGN
    qwen35_benchmark_protocol.py summarize CAMPAIGN
    qwen35_benchmark_protocol.py judge CAMPAIGN --record RECORD --candidate LABEL [--reference-binary LABEL]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable


# --- Protocol constants --------------------------------------------------------

#: The two benchmark tests, each run in its own process.
BENCHMARKS = {
    "baseline": "qwen::qwen35_metal_baseline_benchmark",
    "public": "qwen::qwen35_metal_public_causal_greedy_benchmark",
}
#: Report ``benchmark`` field each test writes.
REPORT_NAMES = {"baseline": "qwen35_08b_f16_baseline", "public": "qwen35_08b_f16_public_causal_greedy"}
#: Source lengths and warm metrics of the baseline benchmark.
SOURCE_LENGTHS = (32, 128, 512)
WARM_METRICS = ("prefill", "first_token_latency", "decode_model_only", "decode_including_greedy_readback")
#: The diagnostic metric: one observation per process, including lazy kernel compilation.
DIAGNOSTIC_METRICS = ("cold_first_token",)
#: Rounds come in Latin-square blocks of four.
BLOCK_ROUNDS = 4
MINIMUM_ROUNDS = 8
MAXIMUM_ROUNDS = 16
#: Seconds since the previous process ended before a sample may start.
COOLDOWN_SECONDS = 120
#: Highest 1-minute load average at which a sample may start.
LOAD_THRESHOLD = 4.0
#: Highest 5-minute load average at which a sample may start (sustained quiet, not a lull).
SUSTAINED_LOAD_THRESHOLD = 6.0
LOAD_POLL_SECONDS = 15
#: In-run 1-minute load above which other work contaminated the sample: the admission
#: threshold plus the benchmark's own two runnable threads (one dispatch, one driver).
IN_RUN_LOAD_LIMIT = LOAD_THRESHOLD + 2.0
IN_RUN_POLL_SECONDS = 5
#: Attempts per planned sample before the campaign stops (resumably).
MAXIMUM_ATTEMPTS = 4
#: The runner stops (resumably) when the load stays above the threshold this long.
MAXIMUM_LOAD_WAIT_SECONDS = 3600
#: A process that runs longer is killed and the campaign stops.
SAMPLE_TIMEOUT_SECONDS = 1500
#: Steady state: the shortest window (two whole blocks), the practical drift tolerance of a
#: block's gated median against the later rounds' median, and the number of standard errors
#: (the conventional three-sigma control limit) by which a drift must also exceed the noise.
STEADY_WINDOW_ROUNDS = 2 * BLOCK_ROUNDS
STEADY_DRIFT = 0.025
STEADY_Z = 3.0
#: The upper quartile of the standard normal distribution: a normal sample's median absolute
#: deviation is this many standard deviations, and the median absolute difference of two
#: independent draws is this many times sqrt(2) standard deviations.
NORMAL_QUARTILE = statistics.NormalDist().inv_cdf(0.75)
#: Judging thresholds: an alert above 5% change, a failure candidate above 10% slowdown.
ALERT = 0.05
FAIL = 0.10
#: Volumes the campaign requires; a missing one stops it.
VOLUMES = ("/Volumes/T9", "/Volumes/bag of holding")
#: The eight round layouts: (first binary index, first binary's order, second binary's order),
#: where "bp" runs the baseline benchmark first and "pb" the public one. Layouts 1-4 and 5-8
#: are each a 4 x 4 Latin square over the (binary, benchmark) cells.
LAYOUTS = (
    (0, "bp", "bp"), (1, "pb", "pb"), (0, "pb", "pb"), (1, "bp", "bp"),
    (0, "bp", "pb"), (1, "pb", "bp"), (0, "pb", "bp"), (1, "bp", "pb"),
)
ORDER = {"b": "baseline", "p": "public"}


class ProtocolError(RuntimeError):
    """A campaign, report, or comparison that the protocol cannot accept."""


# --- Small helpers -------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Hexadecimal SHA-256 of one file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def median(values: Iterable[float]) -> float:
    """Median of a nonempty sequence (the mean of the middle two for an even count)."""
    values = list(values)
    if not values:
        raise ProtocolError("median of no values")
    return statistics.median(values)


def ranks(values: list[float]) -> list[float]:
    """1-based ranks with ties sharing their average rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for index in order[start:end + 1]:
            result[index] = (start + end) / 2 + 1
        start = end + 1
    return result


def spearman(first: list[float], second: list[float]) -> float | None:
    """Spearman rank correlation, or ``None`` when either side is constant or too short."""
    if len(first) != len(second):
        raise ProtocolError("rank correlation needs paired values")
    if len(first) < 3:
        return None
    a, b = ranks(first), ranks(second)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    norm = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return None if norm == 0 else covariance / norm


# --- Plan ----------------------------------------------------------------------

def round_order(number: int) -> list[tuple[int, str]]:
    """The four (binary index, benchmark) runs of 1-based round ``number``, in execution order."""
    if number < 1:
        raise ProtocolError(f"round numbers start at 1, not {number}")
    first, first_order, second_order = LAYOUTS[(number - 1) % len(LAYOUTS)]
    second = 1 - first
    return [(first, ORDER[first_order[0]]), (second, ORDER[second_order[0]]),
            (first, ORDER[first_order[1]]), (second, ORDER[second_order[1]])]


def validate_rounds(rounds: int) -> None:
    """Accept only whole Latin-square blocks between the minimum and maximum round counts."""
    if rounds < MINIMUM_ROUNDS or rounds > MAXIMUM_ROUNDS or rounds % BLOCK_ROUNDS:
        raise ProtocolError(f"a campaign takes {MINIMUM_ROUNDS} to {MAXIMUM_ROUNDS} rounds in blocks of "
                            f"{BLOCK_ROUNDS}, not {rounds}")


def make_plan(rounds: int, binaries: list[dict], model_dir: str, reference_dir: str) -> list[dict]:
    """Every sample of a campaign in execution order, with paths relative to the campaign directory.

    ``binaries`` holds exactly two ``{"label", "source", "binary", "sha256"}`` entries.
    """
    validate_rounds(rounds)
    if len(binaries) != 2 or binaries[0]["label"] == binaries[1]["label"]:
        raise ProtocolError("a campaign compares exactly two distinctly labelled binaries")
    plan = []
    for number in range(1, rounds + 1):
        for position, (index, kind) in enumerate(round_order(number), start=1):
            binary = binaries[index]
            stem = f"r{number:02d}.p{position}.{binary['label']}.{kind}"
            plan.append({
                "index": len(plan), "round": number, "position": position, "label": binary["label"],
                "benchmark": kind, "test": BENCHMARKS[kind], "source": binary["source"],
                "binary": binary["binary"], "sha256": binary["sha256"],
                "argv": [binary["binary"], "--ignored", "--exact", "--nocapture", "--test-threads", "1",
                         BENCHMARKS[kind]],
                "artifacts": {"TRANSFORMERS_QWEN35_DIR": model_dir,
                              "TRANSFORMERS_QWEN35_REFERENCE_DIR": reference_dir},
                "report": f"bench/{stem}.json", "environment": f"env/{stem}.environment.json",
                "log": f"logs/{stem}.log", "host": f"host/{stem}.host.json",
            })
    return plan


def position_counts(plan: list[dict]) -> dict[tuple[str, str, int], int]:
    """How often each (label, benchmark) cell runs in each round position."""
    counts: dict[tuple[str, str, int], int] = {}
    for sample in plan:
        key = (sample["label"], sample["benchmark"], sample["position"])
        counts[key] = counts.get(key, 0) + 1
    return counts


# --- Host observation and the runner -------------------------------------------

def shell(command: str) -> str:
    """Stdout and stderr of one shell command, stripped of the final newline."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return (result.stdout + result.stderr).rstrip("\n")


def load_averages(text: str) -> list[float]:
    """The 1-, 5- and 15-minute load averages from ``sysctl -n vm.loadavg`` output (``{ a b c }``)."""
    parts = text.replace("{", " ").replace("}", " ").split()
    if len(parts) != 3:
        raise ProtocolError(f"unreadable load average: {text!r}")
    return [float(part) for part in parts]


def metal_cache_directory() -> Path:
    """The per-user Metal shader cache, which Metal grows when it compiles new pipelines."""
    return Path(shell("getconf DARWIN_USER_CACHE_DIR")) / "com.apple.metal"


def metal_cache_state(root: Path) -> dict[str, list[int]]:
    """``[size, mtime]`` of every file below the Metal shader cache (read only)."""
    state = {}
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                info = path.stat()
                state[path.relative_to(root).as_posix()] = [info.st_size, int(info.st_mtime)]
    return state


def cache_growth(before: dict[str, list[int]], after: dict[str, list[int]]) -> dict:
    """Changed shader-cache files and the total byte growth between two states."""
    changed = {}
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        if old != new:
            changed[name] = {"before": old, "after": new,
                             "growth_bytes": (new[0] if new else 0) - (old[0] if old else 0)}
    return {"files": changed, "growth_bytes": sum(item["growth_bytes"] for item in changed.values())}


def host_identity() -> dict:
    """Machine and operating-system identity that must not change during a campaign."""
    return {
        "hw_model": shell("sysctl -n hw.model"),
        "cpu": shell("sysctl -n machdep.cpu.brand_string"),
        "performance_cores": int(shell("sysctl -n hw.perflevel0.physicalcpu")),
        "efficiency_cores": int(shell("sysctl -n hw.perflevel1.physicalcpu")),
        "memory_bytes": int(shell("sysctl -n hw.memsize")),
        "os": shell("sw_vers -productName") + " " + shell("sw_vers -productVersion"),
        "os_build": shell("sw_vers -buildVersion"),
        "kernel": shell("uname -v"),
    }


def mounted_volumes() -> set[str]:
    """Mount points among the required volumes that ``mount`` lists."""
    listing = shell("mount")
    return {volume for volume in VOLUMES if f" on {volume} (" in listing}


def host_snapshot(cache: Path) -> dict:
    """Load, processes, power, thermal state, memory pressure, and shader-cache state now."""
    return {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "uptime": shell("uptime"),
        "loadavg": load_averages(shell("sysctl -n vm.loadavg")),
        "top_cpu": shell("ps -Ao pid,pcpu,rss,etime,args -r | head -13 | cut -c1-200"),
        "power": shell("pmset -g batt | head -2"),
        "thermal": shell("pmset -g therm"),
        "memory_pressure": shell("memory_pressure -Q 2>/dev/null | tail -1"),
        "metal_cache": metal_cache_state(cache),
    }


def wait_for_host(previous_end: float, *, cooldown: float = COOLDOWN_SECONDS, threshold: float = LOAD_THRESHOLD,
                  sustained: float = SUSTAINED_LOAD_THRESHOLD, poll: float = LOAD_POLL_SECONDS,
                  maximum_wait: float = MAXIMUM_LOAD_WAIT_SECONDS,
                  clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                  load: Callable[[], list[float]] | None = None) -> dict:
    """Sleep out the cooldown, then poll until the 1- and 5-minute load averages admit a sample.

    A sample starts when the 1-minute average is at most ``threshold`` and the
    5-minute average at most ``sustained``. Returns the cooldown and load wait
    in seconds and the load averages that admitted the sample; raises
    ``ProtocolError`` when the load stays too high for ``maximum_wait``
    seconds after the cooldown.
    """
    load = load or (lambda: load_averages(shell("sysctl -n vm.loadavg")))
    remaining = previous_end + cooldown - clock()
    if remaining > 0:
        sleep(remaining)
    cooled = clock()
    while True:
        current = load()
        if current[0] <= threshold and current[1] <= sustained:
            return {"cooldown_seconds": round(cooled - previous_end, 3),
                    "load_wait_seconds": round(clock() - cooled, 3), "admitting_load_1m": current[0],
                    "admitting_load_5m": current[1]}
        if clock() - cooled >= maximum_wait:
            raise ProtocolError(f"the load averages stayed above {threshold} (1 minute) or {sustained} (5 minutes) "
                                f"for {maximum_wait} s (last {current}); the campaign stops and can be resumed")
        sleep(poll)


def contaminated(series: list[list[float]], limit: float = IN_RUN_LOAD_LIMIT) -> bool:
    """Whether the in-run 1-minute load (``[[seconds, load], ...]``) ever exceeded ``limit``."""
    return any(load > limit for _, load in series)


def write_environment(path: Path, sample: dict, identity: dict, snapshot: dict, gpu: str) -> None:
    """The metadata JSON the benchmark harness embeds in its report."""
    environment = {
        "machine_model": identity["hw_model"],
        "cpu": f"{identity['cpu']} ({identity['performance_cores']} performance + "
               f"{identity['efficiency_cores']} efficiency cores)",
        "accelerator": gpu,
        "physical_memory_bytes": identity["memory_bytes"],
        "operating_system": f"{identity['os']} ({identity['os_build']})",
        "driver_runtime": f"OS-provided Metal ({identity['os']} {identity['os_build']})",
        "power_mode": snapshot["power"].replace("\n", " ") + "; " + shell("pmset -g | grep powermode").strip(),
        "accelerator_exclusive": True,
        "other_workloads": ("shared host: other sessions' processes may run; the heavy-model lock is held for the "
                            f"whole campaign; load averages at admission {snapshot['loadavg']} (thresholds "
                            f"{LOAD_THRESHOLD} for 1 minute, {SUSTAINED_LOAD_THRESHOLD} for 5 minutes; a run whose "
                            f"1-minute load exceeds {IN_RUN_LOAD_LIMIT} is discarded and repeated)"),
        "command": " ".join(["/usr/bin/time", "-l", *sample["argv"]]),
    }
    path.write_text(json.dumps(environment, indent=1) + "\n")


def run_sample(root: Path, sample: dict, identity: dict, gpu: str, previous_end: float) -> tuple[bool, float]:
    """Check the host, wait for it, run one sample under ``/usr/bin/time -l``, and record it.

    Returns whether the attempt was clean (not contaminated by other work) and
    the monotonic end time. A contaminated attempt's report, log, and host
    record are kept under ``contaminated`` names; a failed one raises.
    """
    if mounted_volumes() != set(VOLUMES):
        raise ProtocolError(f"a required volume is not mounted: {sorted(set(VOLUMES) - mounted_volumes())}")
    if host_identity() != identity:
        raise ProtocolError("the host or OS identity changed during the campaign")
    binary = Path(sample["binary"])
    before_digest = sha256_file(binary)
    if before_digest != sample["sha256"]:
        raise ProtocolError(f"binary digest changed: {binary} is {before_digest}")
    admission = wait_for_host(previous_end)
    cache = metal_cache_directory()
    before = host_snapshot(cache)
    if "AC Power" not in before["power"].splitlines()[0]:
        raise ProtocolError(f"not on AC power: {before['power'].splitlines()[0]}")
    for key in ("report", "environment", "log", "host"):
        (root / sample[key]).parent.mkdir(parents=True, exist_ok=True)
    write_environment(root / sample["environment"], sample, identity, before, gpu)
    env = dict(os.environ)
    env.update(sample["artifacts"])
    env["TRANSFORMERS_BENCH_METADATA_JSON"] = str(root / sample["environment"])
    env["TRANSFORMERS_BENCH_OUTPUT_JSON"] = str(root / sample["report"])
    log = root / sample["log"]
    with log.open("w") as handle:
        handle.write(f"== sample {sample['index']} round {sample['round']} position {sample['position']} "
                     f"{sample['label']} {sample['benchmark']}\n")
        handle.write(f"== source {sample['source']}\n")
        handle.write(f"== binary {binary} sha256 {before_digest}\n")
        for key in sorted(sample["artifacts"]):
            handle.write(f"== env {key}={sample['artifacts'][key]}\n")
        handle.write(f"== cwd {root}\n== cmd /usr/bin/time -l {' '.join(sample['argv'])}\n")
        handle.write(f"== admission cooldown={admission['cooldown_seconds']}s "
                     f"load_wait={admission['load_wait_seconds']}s load_1m={admission['admitting_load_1m']} "
                     f"load_5m={admission['admitting_load_5m']}\n")
        handle.write(f"== start {before['utc']} loadavg {before['loadavg']}\n")
        handle.write("== top processes by CPU at start\n" + before["top_cpu"] + "\n")
        handle.write(f"== power {before['power']}\n== thermal {before['thermal']}\n")
        handle.write(f"== memory {before['memory_pressure']}\n\n")
        handle.flush()
        started = time.monotonic()
        in_run: list[list[float]] = []
        process = subprocess.Popen(["/usr/bin/time", "-l", *sample["argv"]], cwd=root, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
        status: int | str | None = None
        while status is None:
            try:
                status = process.wait(timeout=IN_RUN_POLL_SECONDS)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                in_run.append([round(elapsed, 1), load_averages(shell("sysctl -n vm.loadavg"))[0]])
                if elapsed > SAMPLE_TIMEOUT_SECONDS:
                    process.kill()
                    process.wait()
                    status = "timeout"
        ended = time.monotonic()
        after = host_snapshot(cache)
        after_digest = sha256_file(binary)
        growth = cache_growth(before["metal_cache"], after["metal_cache"])
        handle.write(f"\n== end {after['utc']} loadavg {after['loadavg']}\n")
        handle.write("== top processes by CPU at end\n" + after["top_cpu"] + "\n")
        handle.write(f"== metal shader cache growth {growth['growth_bytes']} bytes {json.dumps(growth['files'])}\n")
        handle.write(f"== binary sha256 after {after_digest}\n")
        peak = max((load for _, load in in_run), default=None)
        clean = not contaminated(in_run)
        handle.write(f"== in-run load_1m maximum {peak} over {len(in_run)} observations; limit {IN_RUN_LOAD_LIMIT}; "
                     f"{'clean' if clean else 'CONTAMINATED'}\n")
        handle.write(f"== exit={status} wall={ended - started:.2f}s\n")
    record = {"sample": sample, "exit": status, "wall_seconds": round(ended - started, 3),
              "admission": admission, "binary_sha256_before": before_digest, "binary_sha256_after": after_digest,
              "host_before": before, "host_after": after, "metal_cache_growth": growth,
              "in_run_load_1m": in_run, "in_run_load_1m_maximum": peak, "in_run_load_limit": IN_RUN_LOAD_LIMIT,
              "clean": clean}
    if status != 0 or after_digest != before_digest:
        # Keep the failed attempt as evidence under new names, so that a resumed
        # campaign reruns the sample and the harness's create-only report path is free.
        stamp = before["utc"].replace(":", "")
        for key in ("report", "log"):
            path = root / sample[key]
            if path.exists():
                path.rename(path.with_name(f"{path.stem}.failed-{stamp}{path.suffix}"))
        failed = root / sample["host"]
        failed.with_name(f"{failed.name.removesuffix('.host.json')}.failed-{stamp}.host.json").write_text(
            json.dumps(record, indent=1) + "\n")
        raise ProtocolError(f"sample {sample['index']} ({sample['log']}) exited {status}"
                            + ("" if after_digest == before_digest else " and its binary changed"))
    if not clean:
        stamp = before["utc"].replace(":", "")
        for key in ("report", "log"):
            path = root / sample[key]
            path.rename(path.with_name(f"{path.stem}.contaminated-{stamp}{path.suffix}"))
        host = root / sample["host"]
        host.with_name(f"{host.name.removesuffix('.host.json')}.contaminated-{stamp}.host.json").write_text(
            json.dumps(record, indent=1) + "\n")
        return False, ended
    (root / sample["host"]).write_text(json.dumps(record, indent=1) + "\n")
    return True, ended


def run_campaign(root: Path) -> None:
    """Run every planned sample that has no host record yet, in plan order."""
    campaign = json.loads((root / "plan.json").read_text())
    identity, gpu = campaign["host_identity"], campaign["accelerator"]
    previous_end = time.monotonic()
    for sample in campaign["samples"]:
        if (root / sample["host"]).exists():
            print(f"skip (done) {sample['log']}", flush=True)
            continue
        for attempt in range(1, MAXIMUM_ATTEMPTS + 1):
            print(f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')} start "
                  f"{sample['log']} attempt {attempt}", flush=True)
            clean, previous_end = run_sample(root, sample, identity, gpu, previous_end)
            print(f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')} "
                  f"{'done' if clean else 'contaminated'} {sample['log']} attempt {attempt}", flush=True)
            if clean:
                break
        else:
            raise ProtocolError(f"{sample['log']} was contaminated by other work in {MAXIMUM_ATTEMPTS} attempts; "
                                "the campaign stops and can be resumed")


# --- Per-run metrics -----------------------------------------------------------

def gated_metric_names(kind: str) -> list[str]:
    """Gated metric names of one benchmark, in report order."""
    names = ["strict_load"]
    if kind == "baseline":
        names += [f"source_{length}.{metric}" for length in SOURCE_LENGTHS for metric in WARM_METRICS]
    return names + ["natural_loop"]


def distribution(summary: dict, field: str) -> dict:
    """Recompute one harness distribution's median from its raw samples and check the stored value."""
    samples = summary["samples_seconds"]
    if len(samples) != 10:
        raise ProtocolError(f"{field} has {len(samples)} measured repetitions, not 10")
    value = median(samples)
    if abs(value - summary["median_seconds"]) > 1e-12 * max(1.0, abs(value)):
        raise ProtocolError(f"{field} median {summary['median_seconds']} disagrees with its samples ({value})")
    return {"median": value, "samples": samples}


def run_metrics(report: dict, kind: str) -> dict:
    """Gated per-run values (medians of ten repetitions, or single observations) and the diagnostic."""
    if report.get("benchmark") != REPORT_NAMES[kind]:
        raise ProtocolError(f"expected a {REPORT_NAMES[kind]} report, found {report.get('benchmark')}")
    if report.get("process_memory_instrumentation_enabled") or not report.get("timing_comparable_to_uninstrumented_series"):
        raise ProtocolError("an instrumented report is not timing-comparable")
    gated = {"strict_load": {"median": report["strict_load_seconds"], "samples": [report["strict_load_seconds"]]}}
    if kind == "baseline":
        series = {item["source_length"]: item for item in report["warm_series"]}
        if sorted(series) != list(SOURCE_LENGTHS):
            raise ProtocolError(f"warm series lengths {sorted(series)} are not {list(SOURCE_LENGTHS)}")
        for length in SOURCE_LENGTHS:
            for metric in WARM_METRICS:
                gated[f"source_{length}.{metric}"] = distribution(series[length][metric], f"source {length} {metric}")
    gated["natural_loop"] = distribution(report["natural_greedy"]["total_time"], "natural loop")
    return {"gated": gated, "diagnostic": {"cold_first_token": report["cold_first_token"]["seconds"]}}


def token_identity(report: dict, kind: str) -> dict:
    """Exact token hashes of one report: cold, natural loop, and each warm series."""
    natural = report["natural_greedy"]
    identity = {"cold": report["cold_first_token"]["tokens_u32le_sha256"],
                "natural": natural.get("tokens_u32le_sha256", natural.get("expected_tokens_u32le_sha256"))}
    if kind == "baseline":
        for series in report["warm_series"]:
            hashes = set(series["per_repetition_prediction_hashes"])
            if hashes != {series["predicted_tokens_u32le_sha256"]}:
                raise ProtocolError(f"source {series['source_length']} predictions differ between repetitions")
            identity[f"source_{series['source_length']}"] = series["predicted_tokens_u32le_sha256"]
    return identity


def timing_protocol(report: dict) -> dict:
    """The report's timing protocol without the serialized generation configuration.

    The public benchmark serializes the whole ``GenerationConfig``, whose
    schema grows with the API (later sources add neutral fields such as
    ``logit_bias: {}``); only its strategy, length bound, and stop tokens
    define the timed work.
    """
    protocol = {key: value for key, value in report["protocol"].items() if key != "generation"}
    generation = report["protocol"].get("generation")
    if generation is not None:
        protocol["generation"] = {key: generation[key] for key in ("strategy", "max_new_tokens", "eos_token_ids")}
    return protocol


def compiler(report: dict) -> tuple[str, str]:
    """The ``rustc -vV`` text and the rustup toolchain that compiled a report's executable.

    The compiler is part of a benchmark series: another rustc or LLVM changes
    the generated code and with it the timings, so a report that does not
    record both cannot be placed in a series.
    """
    provenance = report["provenance"]
    rustc = (provenance.get("rustc") or {}).get("value")
    toolchain = provenance.get("compiled_rustup_toolchain")
    if not rustc or not toolchain:
        raise ProtocolError("the report does not record the rustc version and rustup toolchain that built it; "
                            "the compiler is part of the benchmark series")
    return rustc, toolchain


def series_identity(report: dict, kind: str) -> dict:
    """Fields that define a benchmark series: timing protocol, compiler, features, profile, checkpoint, host, and OS."""
    provenance = report["provenance"]
    rustc, toolchain = compiler(report)
    return {"benchmark": kind, "protocol": timing_protocol(report), "rustc": rustc, "rustup_toolchain": toolchain,
            "cargo_features": sorted(provenance["cargo_features"]), "profile": provenance["profile"],
            "backend_profile": provenance["backend_profile"], "repository": provenance["repository"],
            "revision": provenance["revision"], "machine_model": provenance["environment"]["machine_model"],
            "operating_system": provenance["environment"]["operating_system"]}


def binary_identity(report: dict) -> dict:
    """The executable that wrote a report and how it was built: digest, source, compiler, features, and profile.

    A controlled rerun must run the reference binary: either this exact
    executable (``sha256``) or a build of the same ``source_commit`` from a
    clean tree (``worktree_status`` empty) with the same compiler, features,
    and profile. The commit and the clean status together fix the source tree.
    """
    provenance = report["provenance"]
    rustc, toolchain = compiler(report)
    commit = (provenance.get("crate_commit") or {}).get("value")
    status = (provenance.get("crate_worktree_status") or {}).get("value")
    if not commit or status is None:
        raise ProtocolError("the report does not record the source commit and tree status of its executable")
    return {"sha256": provenance["benchmark_executable_sha256"], "source_commit": commit, "worktree_status": status,
            "rustc": rustc, "rustup_toolchain": toolchain, "cargo_features": sorted(provenance["cargo_features"]),
            "profile": provenance["profile"]}


# --- Campaign summary and steady state -----------------------------------------

def robust_noise(values: list[float]) -> dict:
    """Relative round-to-round noise of a series: the smaller of two robust standard deviations.

    Both estimates are consistent for the standard deviation of normal noise
    and are divided by the series median, so they are relative:

    * ``mad``: the median absolute deviation from the median over
      ``NORMAL_QUARTILE``; one outlying round cannot inflate it;
    * ``successive``: the median absolute difference between consecutive
      rounds over ``NORMAL_QUARTILE * sqrt(2)``; a trend or a step among the
      values cannot inflate it, since it changes consecutive rounds little.

    ``noise`` is the smaller one, so neither an outlier nor a trend can widen
    a drift limit; the choice errs toward finding drift, that is, toward more
    rounds.
    """
    if len(values) < 2:
        raise ProtocolError("robust noise needs at least two values")
    centre = median(values)
    if not centre > 0:
        raise ProtocolError(f"robust noise of a series with nonpositive median {centre}")
    mad = median(abs(value - centre) for value in values) / NORMAL_QUARTILE / centre
    successive = median(abs(later - earlier) for earlier, later in zip(values, values[1:])) / (
        NORMAL_QUARTILE * math.sqrt(2.0)) / centre
    return {"mad": mad, "successive": successive, "noise": min(mad, successive)}


def block_drift(block: list[float], later: list[float]) -> dict:
    """Whether a block's median differs from the later rounds' median beyond tolerance and noise.

    ``drift = median(later) / median(block) - 1``, positive when the later
    rounds are slower. With ``s`` the relative noise of the later rounds
    alone (:func:`robust_noise`; the backward scan of :func:`steady_state`
    has already accepted them, starting from the last block as its
    reference), the large-sample standard error of the
    log ratio of the two medians is ``sqrt(pi / 2) * s * sqrt(1 / len(block)
    + 1 / len(later))``, and ``z`` is ``|log(1 + drift)|`` in those units. The
    block drifts when ``|drift| > STEADY_DRIFT`` and ``z > STEADY_Z``: the
    first condition keeps changes too small to matter at a 5% alert from
    counting, the second keeps one-observation-per-process noise (strict load)
    from counting. With zero noise ``z`` is ``None`` and the tolerance alone
    decides.
    """
    if len(block) < 1 or len(later) < 2:
        raise ProtocolError("a block drift needs a nonempty block and at least two later rounds")
    reference, value = median(block), median(later)
    if not reference > 0:
        raise ProtocolError(f"block drift of a series with nonpositive median {reference}")
    drift = value / reference - 1.0
    noise = robust_noise(later)["noise"]
    error = math.sqrt(math.pi / 2.0) * noise * math.sqrt(1.0 / len(block) + 1.0 / len(later))
    shift = abs(math.log(value / reference))
    z = shift / error if error > 0 else None
    drifting = abs(drift) > STEADY_DRIFT and (z is None or z > STEADY_Z)
    return {"drift": drift, "noise": noise, "standard_error": error, "z": z, "drifting": drifting}


def steady_state(values: dict[tuple[str, str, str], list[float]], rounds: int) -> dict:
    """The longest suffix of whole blocks, at least ``STEADY_WINDOW_ROUNDS`` long, in which no block drifts.

    ``values`` maps (label, benchmark, metric) to one value per round, in
    round order; ``rounds`` is a whole number of ``BLOCK_ROUNDS`` blocks. Each
    candidate window starts at a block boundary and spans at least
    ``STEADY_WINDOW_ROUNDS`` rounds; its first block is compared with all
    later rounds by :func:`block_drift`, series by series. The window starts at
    the earliest candidate from which no candidate block drifts, so the rule
    is monotone: every later window passed too, and one lucky window cannot
    stand in for a drift that shows up later. Because windows are whole
    Latin-square blocks, every cell runs equally often in every position
    inside them.

    Returns the first round and the rounds of the window (``None`` and ``[]``
    without one); every candidate block's assessment; the latest drifting
    block (``limiting_block``, which ends the window or, when it is the last
    candidate, leaves the campaign unsteady) and its drifting series; and the
    drifts of the chosen window's first block (or of the limiting block, when
    there is no window) with their largest magnitude, how many of them exceed
    the tolerance, and the largest ``z`` among those.
    """
    if rounds < STEADY_WINDOW_ROUNDS or rounds % BLOCK_ROUNDS:
        raise ProtocolError(f"steady state is assessed over whole blocks of {BLOCK_ROUNDS} rounds and at least "
                            f"{STEADY_WINDOW_ROUNDS} rounds, not {rounds}")
    for key, series in values.items():
        if len(series) != rounds:
            raise ProtocolError(f"{key} has {len(series)} rounds, not {rounds}")
    assessments = []
    for first in range(1, rounds - STEADY_WINDOW_ROUNDS + 2, BLOCK_ROUNDS):
        end = first - 1 + BLOCK_ROUNDS
        series = {"/".join(key): block_drift(per_round[first - 1:end], per_round[end:])
                  for key, per_round in values.items()}
        assessments.append({"first_round": first, "block": list(range(first, end + 1)),
                            "later_rounds": list(range(end + 1, rounds + 1)), "series": series,
                            "drifting": [name for name, item in series.items() if item["drifting"]]})
    chosen = None
    for assessment in reversed(assessments):
        if assessment["drifting"]:
            break
        chosen = assessment
    limiting = next((assessment for assessment in reversed(assessments) if assessment["drifting"]), None)
    shown = chosen or limiting
    first = chosen["first_round"] if chosen else None
    beyond = [item for item in shown["series"].values() if abs(item["drift"]) > STEADY_DRIFT]
    return {"first_round": first, "rounds": list(range(first, rounds + 1)) if first else [],
            "assessments": assessments,
            "limiting_block": limiting["first_round"] if limiting else None,
            "drifting": limiting["drifting"] if limiting else [],
            "drift": {name: item["drift"] for name, item in shown["series"].items()},
            "maximum_absolute_drift": max(abs(item["drift"]) for item in shown["series"].values()),
            "beyond_tolerance": len(beyond),
            "maximum_z_beyond_tolerance": max((item["z"] for item in beyond if item["z"] is not None), default=None)}


def statistics_of(values: list[float]) -> dict | None:
    """Median, minimum, maximum, and the values themselves; ``None`` for no values."""
    if not values:
        return None
    return {"median": median(values), "minimum": min(values), "maximum": max(values), "values": values}


def cold_diagnostic(runs: list[dict], window: list[int]) -> dict:
    """The cold-first-token diagnostic of one cell: statistics, rank correlations, and shader-cache exposure.

    Cold first token includes Metal pipeline compilation, which every process
    of a binary shortens for the next by growing the per-user shader cache.
    A discarded (contaminated or failed) attempt runs the same binary and
    benchmark just before the sample it is repeated for, so that sample
    starts with a warmer cache than the balanced design gives it. The
    diagnostic therefore lists each round's discarded attempts' cache growth
    and repeats the statistics and correlations without those rounds, and it
    correlates cold time with the growth of the sample and its discarded
    attempts together.
    """
    cold = [run["metrics"]["diagnostic"]["cold_first_token"] for run in runs]
    growth = [run["metal_cache_growth_bytes"] for run in runs]
    attempts = [sum(attempt["metal_cache_growth_bytes"] for attempt in run["discarded_attempts"]) for run in runs]
    rounds = [float(run["round"]) for run in runs]
    first = [index for index, run in enumerate(runs) if not run["discarded_attempts"]]
    in_window = [index for index, run in enumerate(runs) if run["round"] in window]
    diagnostic = {
        "gated": False,
        "all_rounds": statistics_of(cold),
        "metal_cache_growth_bytes": growth,
        "spearman_with_metal_cache_growth": spearman(cold, [float(value) for value in growth]),
        "spearman_with_round": spearman(cold, rounds),
        "discarded_attempts_metal_cache_growth_bytes": attempts,
        "rounds_after_discarded_attempts": [run["round"] for run in runs if run["discarded_attempts"]],
        "spearman_with_metal_cache_growth_including_discarded_attempts": spearman(
            cold, [float(own + earlier) for own, earlier in zip(growth, attempts)]),
        "without_rounds_after_discarded_attempts": {
            "all_rounds": statistics_of([cold[index] for index in first]),
            "spearman_with_metal_cache_growth": spearman([cold[index] for index in first],
                                                         [float(growth[index]) for index in first]),
            "spearman_with_round": spearman([cold[index] for index in first], [rounds[index] for index in first]),
        },
    }
    if window:
        diagnostic["steady_state"] = statistics_of([cold[index] for index in in_window])
        diagnostic["without_rounds_after_discarded_attempts"]["steady_state"] = statistics_of(
            [cold[index] for index in in_window if index in first])
    return diagnostic


def summarize(samples: list[dict]) -> dict:
    """Steady state, gated statistics, binaries, and the cold-first-token diagnostic of one campaign.

    Each gated statistic is the median, range, and per-round values over the
    steady-state rounds, with their relative round-to-round ``noise``
    (:func:`robust_noise`), which bounds how finely the window can resolve a
    drift: three standard errors of a block against four later rounds are
    about 2.66 times it.

    Each sample holds ``round``, ``position``, ``label``, ``benchmark``,
    ``metrics`` (from :func:`run_metrics`), ``tokens`` (from
    :func:`token_identity`), ``binary`` (from :func:`binary_identity`),
    ``metal_cache_growth_bytes``, and ``discarded_attempts`` (the contaminated
    or failed attempts run for it first, each with its shader-cache growth).
    """
    rounds = max(sample["round"] for sample in samples)
    labels = sorted({sample["label"] for sample in samples})
    cells = {(sample["label"], sample["benchmark"]): [] for sample in samples}
    for sample in samples:
        cells[(sample["label"], sample["benchmark"])].append(sample)
    for key, runs in cells.items():
        if sorted(run["round"] for run in runs) != list(range(1, rounds + 1)):
            raise ProtocolError(f"{key} does not have exactly one run per round")
        runs.sort(key=lambda run: run["round"])
        if any(run["tokens"] != runs[0]["tokens"] for run in runs):
            raise ProtocolError(f"{key} token identities differ between rounds")
    binaries = {}
    for sample in samples:
        if binaries.setdefault(sample["label"], sample["binary"]) != sample["binary"]:
            raise ProtocolError(f"the label {sample['label']!r} ran more than one executable or build")
    gated = {(label, kind, metric): [run["metrics"]["gated"][metric]["median"] for run in runs]
             for (label, kind), runs in cells.items() for metric in gated_metric_names(kind)}
    steady = steady_state(gated, rounds)
    window = steady["rounds"]
    steady["position_counts"] = {
        f"{label}/{kind}": {str(position): sum(1 for run in runs if run["round"] in window
                                               and run["position"] == position)
                            for position in sorted({run["position"] for run in runs})}
        for (label, kind), runs in sorted(cells.items())}
    result = {"rounds": rounds, "labels": labels, "binaries": binaries, "steady_state": steady, "cells": {}}
    for (label, kind), runs in sorted(cells.items()):
        chosen = [run for run in runs if run["round"] in window]
        cell = {"gated": {}, "tokens": runs[0]["tokens"],
                "diagnostic": {"cold_first_token": cold_diagnostic(runs, window)}}
        if chosen:
            for metric in gated_metric_names(kind):
                values = [run["metrics"]["gated"][metric]["median"] for run in chosen]
                cell["gated"][metric] = {**statistics_of(values), "noise": robust_noise(values)["noise"]}
        result["cells"][f"{label}/{kind}"] = cell
    return result


def discarded_attempts(root: Path, sample: dict) -> list[dict]:
    """The contaminated or failed attempts the runner set aside before a sample's clean run, in time order.

    The runner keeps each discarded attempt's host record beside the clean
    one as ``STEM.contaminated-STAMP.host.json`` or ``STEM.failed-STAMP.host.json``.
    """
    host = root / sample["host"]
    stem = host.name.removesuffix(".host.json")
    attempts = []
    for path in sorted(host.parent.glob(f"{stem}.*.host.json")):
        kind = path.name[len(stem) + 1:].split("-", 1)[0]
        if kind not in ("contaminated", "failed"):
            continue
        record = json.loads(path.read_text())
        if record["sample"] != sample:
            raise ProtocolError(f"{path.name} records another sample than {sample['host']}")
        attempts.append({"host": path.relative_to(root).as_posix(), "kind": kind,
                         "started_utc": record["host_before"]["utc"],
                         "metal_cache_growth_bytes": record["metal_cache_growth"]["growth_bytes"]})
    return attempts


def load_campaign(root: Path) -> list[dict]:
    """Every completed sample of a campaign with its metrics, tokens, identities, and shader-cache growth."""
    campaign = json.loads((root / "plan.json").read_text())
    samples = []
    for sample in campaign["samples"]:
        host = json.loads((root / sample["host"]).read_text())
        if host["exit"] != 0 or host["binary_sha256_before"] != sample["sha256"]:
            raise ProtocolError(f"{sample['host']} records a failed or unauthenticated run")
        if host.get("clean") is False:
            raise ProtocolError(f"{sample['host']} records a run contaminated by other work")
        report = json.loads((root / sample["report"]).read_text())
        if report["provenance"]["benchmark_executable_sha256"] != sample["sha256"]:
            raise ProtocolError(f"{sample['report']} was written by another executable")
        samples.append({"round": sample["round"], "position": sample["position"], "label": sample["label"],
                        "benchmark": sample["benchmark"], "metrics": run_metrics(report, sample["benchmark"]),
                        "tokens": token_identity(report, sample["benchmark"]),
                        "series": series_identity(report, sample["benchmark"]),
                        "binary": binary_identity(report),
                        "metal_cache_growth_bytes": host["metal_cache_growth"]["growth_bytes"],
                        "discarded_attempts": discarded_attempts(root, sample),
                        "admission": host["admission"]})
    for kind in BENCHMARKS:
        series = {json.dumps(sample["series"], sort_keys=True) for sample in samples if sample["benchmark"] == kind}
        if len(series) != 1:
            raise ProtocolError(f"the campaign mixes {kind} benchmark series "
                                "(protocol, compiler, features, checkpoint, host or OS)")
    return samples


def campaign_series(samples: list[dict]) -> dict:
    """The one series identity of each benchmark in a loaded campaign."""
    return {kind: next(sample["series"] for sample in samples if sample["benchmark"] == kind) for kind in BENCHMARKS}


# --- Judging -------------------------------------------------------------------

def classify(ratio: float) -> str:
    """``ok``, ``alert`` (a change above 5% either way) or ``failure_candidate`` (above 10% slower)."""
    if ratio > 1.0 + FAIL:
        return "failure_candidate"
    if ratio > 1.0 + ALERT or ratio < 1.0 - ALERT:
        return "alert"
    return "ok"


def check_reference_binary(reference: dict, summary: dict, label: str) -> str:
    """Require campaign label ``label`` to be the reference binary, and say how it was matched.

    The reference block records the executable's identity (``binary``). The
    rerun matches by ``sha256`` when it is that executable, or by
    ``source_and_toolchain`` when it is a build of the same source commit from
    a clean tree with the same rustc, rustup toolchain, features, and profile.
    It must also reproduce the reference's tokens in both benchmarks. Anything
    else could pair a slowed candidate with another slow binary and hide a
    real regression, so it is refused.
    """
    recorded = reference.get("binary")
    if not recorded:
        raise ProtocolError("the reference block records no binary identity, so no rerun can be shown to run the "
                            "reference binary; a failure cannot be confirmed against it")
    if label not in summary["binaries"]:
        raise ProtocolError(f"the campaign has no label {label!r} to serve as the reference binary")
    rerun = summary["binaries"][label]
    if rerun["sha256"] == recorded["sha256"]:
        matched = "sha256"
    elif (rerun["worktree_status"] == "" and recorded["worktree_status"] == ""
          and all(rerun[key] == recorded[key]
                  for key in ("source_commit", "rustc", "rustup_toolchain", "cargo_features", "profile"))):
        matched = "source_and_toolchain"
    else:
        raise ProtocolError(
            f"label {label!r} is not the reference binary: its executable {rerun['sha256']} is not the recorded "
            f"{recorded['sha256']}, and it is not a clean build of {recorded['source_commit']} with "
            f"{recorded['rustup_toolchain']} and features {recorded['cargo_features']}")
    for kind in BENCHMARKS:
        if summary["cells"][f"{label}/{kind}"]["tokens"] != reference["benchmarks"][kind]["tokens"]:
            raise ProtocolError(f"the reference binary {label!r} does not reproduce the reference's {kind} tokens")
    return matched


def judge(reference: dict, summary: dict, candidate: str, reference_binary: str | None = None) -> dict:
    """Judge a candidate's steady-state medians against a recorded reference.

    ``reference`` is a re-baseline record's ``reference`` block: its binary
    and series identities, token identities, and per-benchmark gated and
    diagnostic statistics. ``summary`` is :func:`summarize`'s output with the
    campaign's ``series``. With ``reference_binary`` (the label of the
    reference binary rerun in the same campaign, which
    :func:`check_reference_binary` verifies) the campaign is a controlled
    rerun, which alone can confirm a failure.
    """
    if summary["steady_state"]["first_round"] is None:
        raise ProtocolError("the candidate campaign did not reach steady state; add rounds before judging")
    if summary["series"] != reference["series"]:
        raise ProtocolError("the candidate is a different benchmark series (protocol, compiler, features, checkpoint, "
                            "host or OS) from the reference; re-baselining is a maintainer decision")
    if candidate not in summary["binaries"]:
        raise ProtocolError(f"the campaign has no candidate label {candidate!r}")
    matched = None
    if reference_binary is not None:
        if reference_binary == candidate:
            raise ProtocolError("the candidate cannot serve as its own reference binary")
        matched = check_reference_binary(reference, summary, reference_binary)
    verdicts = {}
    for kind in BENCHMARKS:
        cell = summary["cells"][f"{candidate}/{kind}"]
        if cell["tokens"] != reference["benchmarks"][kind]["tokens"]:
            raise ProtocolError(f"the candidate's {kind} tokens differ from the reference; timing is not comparable")
        rerun = summary["cells"][f"{reference_binary}/{kind}"] if reference_binary else None
        for metric in gated_metric_names(kind):
            recorded = reference["benchmarks"][kind]["gated"][metric]["median"]
            value = cell["gated"][metric]["median"]
            ratio = value / recorded
            verdict = {"reference": recorded, "candidate": value, "ratio": ratio, "status": classify(ratio)}
            if rerun is not None:
                concurrent = rerun["gated"][metric]["median"]
                verdict["reference_binary_in_campaign"] = concurrent
                verdict["ratio_to_reference_binary"] = value / concurrent
                verdict["reference_binary_drift"] = concurrent / recorded
                if ratio > 1.0 + FAIL:
                    if value / concurrent > 1.0 + FAIL:
                        verdict["status"] = "fail"
                    elif concurrent / recorded > 1.0 + FAIL:
                        verdict["status"] = "series_drift"
                    else:
                        verdict["status"] = "not_confirmed"
            verdicts[f"{kind}/{metric}"] = verdict
        cold = cell["diagnostic"]["cold_first_token"]
        first_runs = cold["without_rounds_after_discarded_attempts"]["steady_state"]
        verdicts[f"{kind}/cold_first_token"] = {
            "status": "diagnostic", "gated": False,
            "median": cold["steady_state"]["median"],
            "range": [cold["steady_state"]["minimum"], cold["steady_state"]["maximum"]],
            "median_without_rounds_after_discarded_attempts": first_runs["median"] if first_runs else None,
            "reference_median": reference["benchmarks"][kind]["diagnostic"]["cold_first_token"]["median"],
        }
    statuses = {verdict["status"] for verdict in verdicts.values()}
    for overall in ("fail", "series_drift", "failure_candidate", "not_confirmed", "alert"):
        if overall in statuses:
            break
    else:
        overall = "ok"
    return {"candidate": candidate, "controlled_rerun": reference_binary is not None,
            "reference_binary": {"label": reference_binary, "matched_by": matched} if reference_binary else None,
            "overall": overall, "metrics": verdicts}


# --- Command line --------------------------------------------------------------

def campaign_summary(root: Path) -> dict:
    """:func:`summarize` of a campaign directory, with its series identity."""
    samples = load_campaign(root)
    summary = summarize(samples)
    summary["series"] = campaign_series(samples)
    return summary


def reference_block(summary: dict, label: str) -> dict:
    """The ``reference`` block a re-baseline record stores for ``label``: what :func:`judge` reads.

    It holds the label's binary identity (:func:`binary_identity`), so that a
    controlled rerun can be checked to run the reference binary, the series
    identity, the steady-state rounds, and per benchmark the tokens, the gated
    statistics, and the ungated cold-first-token statistics.
    """
    if summary["steady_state"]["first_round"] is None:
        raise ProtocolError("a campaign that did not reach steady state cannot become a reference")
    benchmarks = {}
    for kind in BENCHMARKS:
        cell = summary["cells"][f"{label}/{kind}"]
        diagnostic = cell["diagnostic"]["cold_first_token"]
        cold = {"gated": False, **diagnostic["steady_state"],
                "without_rounds_after_discarded_attempts":
                    diagnostic["without_rounds_after_discarded_attempts"]["steady_state"]}
        benchmarks[kind] = {
            "tokens": cell["tokens"],
            "gated": {metric: {key: value for key, value in statistic.items()}
                      for metric, statistic in cell["gated"].items()},
            "diagnostic": {"cold_first_token": cold},
        }
    return {"label": label, "binary": summary["binaries"][label], "series": summary["series"],
            "steady_state_rounds": summary["steady_state"]["rounds"], "benchmarks": benchmarks}


def main(argv: list[str] | None = None) -> int:
    """Parse a subcommand and run it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="write CAMPAIGN/plan.json")
    plan.add_argument("campaign", type=Path)
    plan.add_argument("--rounds", type=int, default=MINIMUM_ROUNDS)
    plan.add_argument("--model-dir", required=True)
    plan.add_argument("--reference-dir", required=True)
    plan.add_argument("--accelerator", required=True, help="GPU description for the harness metadata")
    plan.add_argument("binaries", nargs=2, metavar="LABEL=SOURCE=BINARY")
    run = commands.add_parser("run", help="run every sample without a host record")
    run.add_argument("campaign", type=Path)
    summary = commands.add_parser("summarize", help="write CAMPAIGN/summary.json")
    summary.add_argument("campaign", type=Path)
    verdict = commands.add_parser("judge", help="judge a candidate against a re-baseline record")
    verdict.add_argument("campaign", type=Path)
    verdict.add_argument("--record", type=Path, required=True)
    verdict.add_argument("--candidate", required=True)
    verdict.add_argument("--reference-binary")
    args = parser.parse_args(argv)
    if args.command == "plan":
        binaries = []
        for item in args.binaries:
            label, rest = item.split("=", 1)
            source, binary = rest.rsplit("=", 1)
            binaries.append({"label": label, "source": source, "binary": binary,
                             "sha256": sha256_file(Path(binary))})
        args.campaign.mkdir(parents=True, exist_ok=True)
        document = {"protocol": "BENCHMARKS.md: Qwen3.5-0.8B benchmark campaign protocol",
                    "script_sha256": sha256_file(Path(__file__)), "rounds": args.rounds,
                    "host_identity": host_identity(), "accelerator": args.accelerator, "binaries": binaries,
                    "constants": {"cooldown_seconds": COOLDOWN_SECONDS, "load_threshold_1m": LOAD_THRESHOLD,
                                  "sustained_load_threshold_5m": SUSTAINED_LOAD_THRESHOLD,
                                  "in_run_load_limit_1m": IN_RUN_LOAD_LIMIT,
                                  "in_run_poll_seconds": IN_RUN_POLL_SECONDS, "maximum_attempts": MAXIMUM_ATTEMPTS,
                                  "load_poll_seconds": LOAD_POLL_SECONDS,
                                  "maximum_load_wait_seconds": MAXIMUM_LOAD_WAIT_SECONDS,
                                  "sample_timeout_seconds": SAMPLE_TIMEOUT_SECONDS,
                                  "steady_window_rounds": STEADY_WINDOW_ROUNDS, "steady_drift": STEADY_DRIFT,
                                  "steady_z": STEADY_Z,
                                  "alert": ALERT, "fail": FAIL},
                    "samples": make_plan(args.rounds, binaries, args.model_dir, args.reference_dir)}
        (args.campaign / "plan.json").write_text(json.dumps(document, indent=1) + "\n")
        print(f"{len(document['samples'])} samples -> {args.campaign / 'plan.json'}")
    elif args.command == "run":
        run_campaign(args.campaign.resolve())
    elif args.command == "summarize":
        (args.campaign / "summary.json").write_text(json.dumps(campaign_summary(args.campaign), indent=1) + "\n")
    else:
        record = json.loads(args.record.read_text())
        print(json.dumps(judge(record["reference"], campaign_summary(args.campaign), args.candidate,
                               args.reference_binary), indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ProtocolError as error:
        print(f"benchmark protocol error: {error}", file=sys.stderr)
        sys.exit(2)
