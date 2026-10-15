"""Sample a supervised macOS benchmark's process memory without changing runtime APIs.

This separate memory run is NOT timing-comparable to normal benchmarks. The child
opts into private Unix-socket phase markers through TRANSFORMERS_BENCH_MEMORY_SOCKET.
For each phase it blocks during a measured idle window, acknowledges begin, executes
and synchronizes its backend work, then acknowledges end. Only one phase is active.

Python's stdlib ctypes calls proc_pid_rusage(RUSAGE_INFO_V4). ABI declarations are
pinned to the locally reviewed Darwin SDK headers, not guessed from process output:
https://developer.apple.com/documentation/kernel/rusage_info_v4
https://github.com/apple-oss-distributions/xnu/blob/main/libsyscall/wrappers/libproc/libproc.h
The SDK headers themselves are not redistributed. Their hashes and exact field
sequence are verified before spawning a child. The process-lifetime identity is
PID plus ri_proc_start_abstime, checked on every successful observation. Executable
UUIDs are retained separately: the same process can change its image during exec.

ri_phys_footprint is OS-accounted process physical footprint on unified memory,
not a host/device split or allocator payload. Per-phase maxima are SAMPLED lower
bounds: scheduler delays and unobserved transients remain possible. Raw JSONL retains
every query's monotonic timestamps, duration, errors and observed counters. Summary
windows retain actual maximum sampling/edge gaps, absolute sampled maxima, measured
idle baseline distributions and signed deltas. Kernel-reported lifetime high-water
is separate and is never relabeled a per-phase maximum. No reset/control API is used.

Example (already-built executable, no Cargo compilation inside the supervised run):
  python3 scripts/profile_process_memory.py --output-dir /private/tmp/new-memory-run \
    -- /absolute/test-binary qwen::qwen35_metal_baseline_benchmark \
    --ignored --exact --nocapture --test-threads=1
The benchmark's ordinary environment arguments remain required. Its own JSON output
must use a separate create-only path. This supervisor also creates its output folder
once and never overwrites evidence. Timeouts terminate only the child it spawned.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import socket
import statistics
import subprocess
import sys
import tempfile
import time


# Reviewed SDK header identities. A header is accepted only when its exact bytes
# match one of these digests; the V4 structure is then re-parsed below.
# resource.h 1067178025... is the macOS 27.0 SDK revision: its rusage_info_v4
# declaration (field order, RUSAGE_INFO_V4 = 4) was compared with the earlier
# reviewed header on 2026-10-15 and is unchanged.
HEADER_HASHES = {
    "usr/include/sys/resource.h": (
        "7d16930e6b75f11ba203238faa5580d31d48fcd4230f2b3f604aaa5fd7e86b58",
        "1067178025c5fd88c9a1b8206a07c2695ef94dd471fe2783f06c512995382961",
    ),
    "usr/include/libproc.h": ("246d87709fc6b9157ce5cf3c475656ac48e0e1ae8bbdc46cf45acd34294448cd",),
}
# All 35 uint64 fields after the 16-byte executable UUID, in SDK declaration order.
U64_FIELDS = (
    "ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins "
    "ri_wired_size ri_resident_size ri_phys_footprint ri_proc_start_abstime "
    "ri_proc_exit_abstime ri_child_user_time ri_child_system_time ri_child_pkg_idle_wkups "
    "ri_child_interrupt_wkups ri_child_pageins ri_child_elapsed_abstime ri_diskio_bytesread "
    "ri_diskio_byteswritten ri_cpu_time_qos_default ri_cpu_time_qos_maintenance "
    "ri_cpu_time_qos_background ri_cpu_time_qos_utility ri_cpu_time_qos_legacy "
    "ri_cpu_time_qos_user_initiated ri_cpu_time_qos_user_interactive ri_billed_system_time "
    "ri_serviced_system_time ri_logical_writes ri_lifetime_max_phys_footprint "
    "ri_instructions ri_cycles ri_billed_energy ri_serviced_energy "
    "ri_interval_max_phys_footprint ri_runnable_time"
).split()
MAX_EVENT_BYTES = 4096
MAX_SAMPLES = 2_000_000


def file_sha256(path: Path) -> str:
    """Hash potentially large raw traces with fixed 64KiB host staging."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RUsageInfoV4(ctypes.Structure):
    """The native, verified 296-byte Darwin V4 resource-usage ABI."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in U64_FIELDS]


def verify_sdk_headers(sdk: Path) -> dict:
    """Fail closed if either the reviewed header identity or ABI declaration changes."""

    sources = {}
    for relative, expected in HEADER_HASHES.items():
        path = sdk / relative
        data = path.read_bytes()
        observed = hashlib.sha256(data).hexdigest()
        if observed not in expected:
            raise ValueError(f"unreviewed SDK header: {path} SHA-256 {observed}")
        sources[relative] = {"sha256": observed, "bytes": len(data)}
    text = (sdk / "usr/include/sys/resource.h").read_text()
    body = re.search(r"struct rusage_info_v4\s*\{([^}]+)\};", text)
    if body is None or not re.search(r"#define\s+RUSAGE_INFO_V4\s+4\b", text):
        raise ValueError("missing V4 declaration")
    fields = re.findall(r"uint64_t\s+(ri_\w+)\s*;", body.group(1))
    if fields != U64_FIELDS or not re.search(r"uint8_t\s+ri_uuid\[16\];", body.group(1)):
        raise ValueError("SDK field ordering changed")
    if (ctypes.sizeof(RUsageInfoV4), RUsageInfoV4.ri_resident_size.offset,
            RUsageInfoV4.ri_phys_footprint.offset, RUsageInfoV4.ri_lifetime_max_phys_footprint.offset) != (296, 64, 72, 240):
        raise ValueError("unsupported native ctypes structure layout")
    return {"sdk": str(sdk.resolve()), "headers": sources, "flavor": 4,
            "structure_bytes": ctypes.sizeof(RUsageInfoV4), "field_order": ["ri_uuid"] + U64_FIELDS}


class DarwinSampler:
    """Read-only libproc adapter; errors retain errno and are never replaced by zero."""

    def __init__(self):
        """Require Darwin and the exact reviewed signature before observing children."""

        if sys.platform != "darwin":
            raise ValueError("this process-memory supervisor requires macOS")
        self.library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.call = self.library.proc_pid_rusage
        self.call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.call.restype = ctypes.c_int

    def sample(self, pid: int) -> dict:
        """Read one coherent kernel record, including process-lifetime identity."""

        record = RUsageInfoV4()
        ctypes.set_errno(0)
        if self.call(pid, 4, ctypes.byref(record)) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return {
            "pid": pid, "process_start_abstime": record.ri_proc_start_abstime,
            "process_exit_abstime": record.ri_proc_exit_abstime,
            "executable_uuid": bytes(record.ri_uuid).hex(),
            "resident_bytes": record.ri_resident_size,
            "physical_footprint_bytes": record.ri_phys_footprint,
            "kernel_lifetime_max_physical_footprint_bytes": record.ri_lifetime_max_phys_footprint,
        }


class Window:
    """Bounded online sample statistics, retaining baseline values only when needed."""

    def __init__(self, start_ns: int, baseline: bool = False):
        """Start a marker-bounded interval without inventing an initial memory reading."""

        self.start_ns = start_ns
        self.baseline = baseline
        self.count = self.errors = 0
        self.first_ns = self.last_ns = None
        self.maximum_gap_ns = 0
        self.maximum = {"resident_bytes": None, "physical_footprint_bytes": None}
        self.values = {name: [] for name in self.maximum}

    def add(self, timestamp_ns: int, sample: dict | None) -> None:
        """Use successful sample gaps; failed probes create visible holes in coverage."""

        if sample is None:
            self.errors += 1
            return
        if timestamp_ns < self.start_ns or (self.last_ns is not None and timestamp_ns < self.last_ns):
            raise ValueError("non-monotonic sample")
        self.maximum_gap_ns = max(self.maximum_gap_ns, timestamp_ns - (self.last_ns if self.last_ns is not None else self.start_ns))
        self.first_ns = timestamp_ns if self.first_ns is None else self.first_ns
        self.last_ns = timestamp_ns
        self.count += 1
        for name in self.maximum:
            value = sample[name]
            if type(value) is not int or value < 0:
                raise ValueError("invalid memory counter")
            self.maximum[name] = value if self.maximum[name] is None else max(self.maximum[name], value)
            if self.baseline:
                self.values[name].append(value)

    def summary(self, end_ns: int) -> dict:
        """Include uncovered interval edges rather than reporting interior gaps alone."""

        if end_ns < self.start_ns or (self.last_ns is not None and end_ns < self.last_ns):
            raise ValueError("window end precedes observation")
        gaps = max(self.maximum_gap_ns, end_ns - (self.last_ns if self.last_ns is not None else self.start_ns))
        result = {"start_ns": self.start_ns, "end_ns": end_ns, "duration_ns": end_ns - self.start_ns,
                  "successful_samples": self.count, "probe_errors": self.errors,
                  "first_successful_sample_ns": self.first_ns, "last_successful_sample_ns": self.last_ns,
                  "mean_successful_sample_interval_ns": (self.last_ns - self.first_ns) / (self.count - 1) if self.count > 1 else None,
                  "maximum_successful_sampling_or_edge_gap_ns": gaps,
                  "sampled_maximum_bytes": self.maximum.copy(), "exact_unsampled_peak": False}
        if self.baseline:
            result["baseline_bytes"] = {name: {"median": statistics.median(values), "minimum": min(values), "maximum": max(values)} if values else None for name, values in self.values.items()}
        return result


class PhaseProtocol:
    """Strict serial phase state machine, independent of subprocess and Darwin APIs."""

    def __init__(self, baseline_ns: int, minimum_baseline_samples: int = 5):
        """Require a genuinely sampled idle window before acknowledging work."""

        self.baseline_ns = baseline_ns
        self.minimum_baseline_samples = minimum_baseline_samples
        self.next_id = 0
        self.current = None
        self.stage = None
        self.phases = []

    def event(self, event: dict, now_ns: int) -> dict | None:
        """Validate identity/order; idle ACK is deferred until enough time and samples."""

        if type(event) is not dict or set(event) != {"schema_version", "id", "event", "phase"} or type(event["schema_version"]) is not int or event["schema_version"] != 1 or type(event["id"]) is not int or event["id"] != self.next_id:
            raise ValueError("invalid phase event identity/schema")
        action, label = event["event"], event["phase"]
        if action == "idle":
            if self.current is not None or not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", label):
                raise ValueError("invalid/nested idle phase")
            self.current = {"id": self.next_id, "name": label, "baseline_window": Window(now_ns, True)}
            self.stage = "idle"
            return None
        if self.current is None:
            raise ValueError("phase begin/end without idle baseline")
        if action == "begin":
            if self.stage != "ready" or label != self.current["name"]:
                raise ValueError("premature or mismatched phase begin")
            self.current["active_window"] = Window(now_ns)
            self.stage = "active"
        elif action == "end":
            if self.stage != "active" or label != "":
                raise ValueError("unmatched phase end")
            baseline = self.current["baseline"]
            observed = self.current["active_window"].summary(now_ns)
            delta = {name: observed["sampled_maximum_bytes"][name] - baseline["baseline_bytes"][name]["median"] if observed["sampled_maximum_bytes"][name] is not None else None for name in observed["sampled_maximum_bytes"]}
            self.phases.append({"id": self.next_id, "name": self.current["name"], "idle_baseline": baseline,
                                "observed": observed, "sampled_maximum_minus_idle_median_bytes": delta,
                                "valid": observed["successful_samples"] > 0 and observed["probe_errors"] == 0 and baseline["probe_errors"] == 0})
            self.current = None
            self.stage = None
            self.next_id += 1
        else:
            raise ValueError("unknown phase event")
        return {"ok": True, "id": event["id"], "event": action}

    def sample(self, now_ns: int, sample: dict | None) -> None:
        """Assign observations only to the corresponding idle or active window."""

        if self.stage in ("idle", "active"):
            self.current["baseline_window" if self.stage == "idle" else "active_window"].add(now_ns, sample)

    def idle_ready(self, now_ns: int) -> dict | None:
        """Release the child after both minimum duration and successful count pass."""

        if self.stage == "idle":
            window = self.current["baseline_window"]
            if now_ns - window.start_ns >= self.baseline_ns and window.count >= self.minimum_baseline_samples:
                self.current["baseline"] = window.summary(now_ns)
                self.stage = "ready"
                return {"ok": True, "id": self.next_id, "event": "idle"}
        return None


def validate_limits(interval_ms: float, baseline_ms: float, timeout_seconds: float) -> None:
    """Cap scheduler requests and process duration before allocating or spawning."""

    for value, minimum, maximum in ((interval_ms, 0.5, 10), (baseline_ms, 5, 1000), (timeout_seconds, 0.05, 7200)):
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError("invalid sampling/baseline/timeout limits")


def supervise(command: list[str], output: Path, interval_ms: float = 1.0,
              baseline_ms: float = 10.0, timeout_seconds: float = 900.0,
              sampler=None, abi: dict | None = None) -> dict:
    """Run one owned child, retaining raw observations and fail-closed phase evidence."""

    validate_limits(interval_ms, baseline_ms, timeout_seconds)
    if not command or any(not isinstance(part, str) or "\0" in part for part in command):
        raise ValueError("an explicit nonempty child argv is required")
    if sampler is None:
        sdk = Path(subprocess.check_output(["xcrun", "--show-sdk-path"], text=True, timeout=10).strip())
        abi = verify_sdk_headers(sdk)
        sampler = DarwinSampler()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    start_ns = time.monotonic_ns()
    protocol = PhaseProtocol(round(baseline_ms * 1_000_000))
    all_samples = Window(start_ns)
    identity = None
    executable_uuids = set()
    errors = []
    counter = 0
    consecutive_errors = 0
    lifetime_max = None
    child = None
    connection = None
    child_code = None
    interval_ns = round(interval_ms * 1_000_000)
    deadline_ns = start_ns + round(timeout_seconds * 1_000_000_000)
    next_sample_ns = start_ns
    raw_path = output / "samples.jsonl"

    with tempfile.TemporaryDirectory(prefix="tfm-memory-", dir="/private/tmp" if sys.platform == "darwin" else "/tmp") as socket_directory, \
            selectors.DefaultSelector() as selector, \
            (output / "stdout.log").open("xb") as stdout, \
            (output / "stderr.log").open("xb") as stderr, \
            raw_path.open("x", buffering=1024 * 1024) as raw, \
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        endpoint = Path(socket_directory) / "phase.sock"
        listener.bind(str(endpoint))
        os.chmod(endpoint, 0o600)
        listener.listen(1)
        listener.setblocking(False)
        selector.register(listener, selectors.EVENT_READ)
        pending = bytearray()

        def acknowledge(ack: dict | None) -> None:
            """Send only protocol ACKs; a disappeared or stalled child is an error."""

            if ack is not None:
                connection.sendall(json.dumps(ack, separators=(",", ":")).encode() + b"\n")

        def collect() -> None:
            """Timestamp one API query, preserving errors and PID-lifetime checks."""

            nonlocal identity, counter, lifetime_max, consecutive_errors
            before = time.monotonic_ns()
            observation = None
            failure = None
            try:
                observation = sampler.sample(child.pid)
                observed_identity = {name: observation[name] for name in ("pid", "process_start_abstime")}
                if observed_identity["process_start_abstime"] == 0:
                    raise ValueError("missing process-lifetime identity")
                if identity is None:
                    identity = observed_identity
                elif identity != observed_identity:
                    raise ValueError("PID/process-lifetime identity changed")
                executable_uuids.add(observation["executable_uuid"])
                value = observation["kernel_lifetime_max_physical_footprint_bytes"]
                lifetime_max = value if lifetime_max is None else max(lifetime_max, value)
                consecutive_errors = 0
            except (OSError, ValueError) as error:
                failure = {"type": type(error).__name__, "message": str(error), "errno": getattr(error, "errno", None)}
                errors.append({"time_ns": before, "phase_id": protocol.current["id"] if protocol.current else None, "query_error": failure})
                observation = None
                consecutive_errors += 1
            after = time.monotonic_ns()
            all_samples.add(after, observation)
            protocol.sample(after, observation)
            raw.write(json.dumps({"query_start_ns": before, "time_ns": after, "query_duration_ns": after - before,
                                  "phase_id": protocol.current["id"] if protocol.current else None,
                                  "stage": protocol.stage, "observation": observation, "error": failure}, separators=(",", ":")) + "\n")
            counter += 1
            if counter >= MAX_SAMPLES or consecutive_errors >= 3:
                raise ValueError("sampling limit or repeated probe failures")
            if failure and failure["message"] == "PID/process-lifetime identity changed":
                raise ValueError(failure["message"])

        try:
            environment = dict(os.environ, TRANSFORMERS_BENCH_MEMORY_SOCKET=str(endpoint))
            child = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr)
            while True:
                now = time.monotonic_ns()
                if now >= deadline_ns:
                    raise TimeoutError("supervised child exceeded its bounded timeout")
                if now >= next_sample_ns:
                    collect()
                    next_sample_ns = time.monotonic_ns() + interval_ns
                acknowledge(protocol.idle_ready(time.monotonic_ns()))
                # Sample before poll reaps the child, permitting a final zombie observation.
                child_code = child.poll()
                if child_code is not None:
                    if protocol.current is not None:
                        raise ValueError("child exited with an unfinished phase")
                    break
                wait = max(0, min(next_sample_ns, deadline_ns) - time.monotonic_ns()) / 1_000_000_000
                for key, _ in selector.select(wait):
                    if key.fileobj is listener:
                        if connection is not None:
                            raise ValueError("multiple phase clients")
                        connection, _ = listener.accept()
                        connection.settimeout(1.0)
                        selector.register(connection, selectors.EVENT_READ)
                    else:
                        data = connection.recv(MAX_EVENT_BYTES + 1)
                        if not data:
                            selector.unregister(connection)
                            connection.close()
                            if protocol.current is not None:
                                raise ValueError("phase socket closed before end")
                            continue
                        pending.extend(data)
                        if len(pending) > MAX_EVENT_BYTES:
                            raise ValueError("oversized phase event")
                        if b"\n" in pending:
                            line, _, remainder = pending.partition(b"\n")
                            if remainder:
                                raise ValueError("pipelined phase events violate ACK ordering")
                            pending.clear()
                            event = json.loads(line)
                            # The end snapshot belongs inside the still-active phase.
                            if isinstance(event, dict) and event.get("event") == "end":
                                collect()
                            ack = protocol.event(event, time.monotonic_ns())
                            if event["event"] == "begin":
                                collect()
                            acknowledge(ack)
        except Exception as error:
            errors.append({"time_ns": time.monotonic_ns(), "supervisor_error": type(error).__name__, "message": str(error)})
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            if child is not None:
                child_code = child.returncode
            if connection is not None:
                connection.close()
            listener.close()
            raw.flush()
            os.fsync(raw.fileno())
    end_ns = time.monotonic_ns()
    validation_failures = []
    if child_code != 0:
        validation_failures.append("child did not exit successfully")
    if errors:
        validation_failures.append("sampling or supervisor errors were recorded")
    if not protocol.phases:
        validation_failures.append("no complete instrumented phases were observed")
    if protocol.current is not None:
        validation_failures.append("child left an unfinished phase")
    if any(not phase["valid"] for phase in protocol.phases):
        validation_failures.append("one or more phases lack valid sampled evidence")
    valid = not validation_failures
    result = {"schema_version": 1, "status": "passed" if valid else "failed", "command": command,
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "platform": sys.platform, "os_release": os.uname().release, "os_version": os.uname().version,
              "architecture": os.uname().machine, "python_executable": sys.executable,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "abi": abi, "identity": identity, "observed_executable_uuids": sorted(executable_uuids), "child_exit_code": child_code,
              "requested_interval_ns": interval_ns, "requested_idle_baseline_ns": protocol.baseline_ns,
              "minimum_idle_samples": protocol.minimum_baseline_samples, "sample_count": counter,
              "whole_observation_window": all_samples.summary(end_ns), "phase_count": len(protocol.phases), "phases": protocol.phases,
              "unfinished_phase": {"id": protocol.current["id"], "name": protocol.current["name"], "stage": protocol.stage} if protocol.current else None,
              "observed_kernel_reported_lifetime_max_physical_footprint_bytes": lifetime_max,
              "kernel_lifetime_counter_is_not_a_phase_peak": True,
              "sampled_maxima_are_exact_unsampled_peaks": False, "host_device_memory_split": None,
              "timing_comparable_to_uninstrumented_series": False,
              "memory_scope": "OS-accounted child-process physical footprint and RSS; unified memory, not allocator payload",
              "sampling_limitations": "samples and IPC-conservative phase brackets can miss transients; actual gaps are retained; no kernel interval counter is reset",
              "errors": errors, "validation_failures": validation_failures,
              "raw_samples": {"relative_path": raw_path.name, "bytes": raw_path.stat().st_size,
                                                           "sha256": file_sha256(raw_path)}}
    with (output / "report.json").open("x") as report:
        json.dump(result, report, separators=(",", ":"), allow_nan=False)
        report.write("\n")
    return result


def main() -> int:
    """Parse an explicit child argv and return nonzero on incomplete memory evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interval-ms", type=float, default=1.0)
    parser.add_argument("--baseline-ms", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        result = supervise(command, args.output_dir, args.interval_ms, args.baseline_ms, args.timeout_seconds)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f"memory supervisor preflight failed: {error}\n")
    print(json.dumps({"status": result["status"], "report": str(args.output_dir / "report.json")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
