"""Bounded supervisor/protocol tests; no model artifacts, Cargo, network or GPU."""

import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "profile_process_memory.py"
SPEC = importlib.util.spec_from_file_location("profile_process_memory", SCRIPT)
MEMORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEMORY)


def observation(pid=123, resident=100, footprint=80):
    """Supply positive process-lifetime identity and separate measured counters."""

    return {"pid": pid, "process_start_abstime": 7, "process_exit_abstime": 0,
            "executable_uuid": "00" * 16, "resident_bytes": resident,
            "physical_footprint_bytes": footprint,
            "kernel_lifetime_max_physical_footprint_bytes": footprint + 1000}


def event(action, label="test", identity=0):
    """Build the exact bounded phase message understood by the Rust helper."""

    return {"schema_version": 1, "id": identity, "event": action, "phase": label}


class FakeSampler:
    """Deterministic process observer for portable child-lifecycle tests."""

    def sample(self, pid):
        """Retain actual child PID while fixing every memory value."""

        return observation(pid)


class MemoryMathTests(unittest.TestCase):
    """Sampling math and phase state cannot overstate their evidence."""

    def test_sampled_baseline_and_edge_gaps(self):
        window = MEMORY.Window(0, True)
        for timestamp, value in [(10, 10), (20, 12), (40, 14)]:
            window.add(timestamp, observation(resident=value, footprint=value + 20))
        window.add(45, None)
        result = window.summary(70)
        self.assertEqual(result["maximum_successful_sampling_or_edge_gap_ns"], 30)
        self.assertEqual(result["baseline_bytes"]["resident_bytes"], {"median": 12, "minimum": 10, "maximum": 14})
        self.assertEqual(result["probe_errors"], 1)
        self.assertFalse(result["exact_unsampled_peak"])
        empty = MEMORY.Window(10).summary(30)
        self.assertEqual(empty["maximum_successful_sampling_or_edge_gap_ns"], 20)
        self.assertIsNone(empty["sampled_maximum_bytes"]["resident_bytes"])
        with self.assertRaises(ValueError):
            window.add(1, observation())
        with self.assertRaises(ValueError):
            window.summary(39)
        with self.assertRaises(ValueError):
            MEMORY.Window(0).add(0, observation(resident=-1))

    def test_phase_handshake_requires_time_and_samples(self):
        protocol = MEMORY.PhaseProtocol(10, 3)
        self.assertIsNone(protocol.event(event("idle"), 0))
        protocol.sample(1, observation())
        protocol.sample(2, observation())
        self.assertIsNone(protocol.idle_ready(100))
        protocol.sample(3, observation())
        self.assertIsNone(protocol.idle_ready(9))
        self.assertEqual(protocol.idle_ready(10), {"ok": True, "id": 0, "event": "idle"})
        protocol.event(event("begin"), 11)
        protocol.sample(12, observation(resident=90, footprint=70))
        protocol.event(event("end", ""), 14)
        phase = protocol.phases[0]
        self.assertTrue(phase["valid"])
        self.assertEqual(phase["sampled_maximum_minus_idle_median_bytes"], {"resident_bytes": -10, "physical_footprint_bytes": -10})
        self.assertEqual(protocol.next_id, 1)

    def test_invalid_phase_schema_order_and_missing_samples(self):
        for malformed in [None, [], {}, event("idle", identity=True), dict(event("idle"), schema_version=True), event("begin"), event("end", ""), event("idle", "space invalid")]:
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                MEMORY.PhaseProtocol(10).event(malformed, 0)
        protocol = MEMORY.PhaseProtocol(10, 1)
        protocol.event(event("idle"), 0)
        for invalid in [event("idle"), event("begin"), event("unknown")]:
            with self.assertRaises(ValueError):
                protocol.event(invalid, 1)
        protocol.sample(1, observation())
        protocol.idle_ready(10)
        with self.assertRaises(ValueError):
            protocol.event(event("begin", "wrong"), 11)
        protocol.event(event("begin"), 11)
        protocol.sample(12, None)
        protocol.event(event("end", ""), 13)
        self.assertFalse(protocol.phases[0]["valid"])
        with self.assertRaises(ValueError):
            protocol.event(event("end", ""), 14)

    def test_limits_and_abi_layout_are_exact(self):
        MEMORY.validate_limits(1, 10, 900)
        for arguments in [(0, 10, 1), (1, 0, 1), (1, 10, 0), (float("nan"), 10, 1), (1, float("inf"), 1), (1, 10, 7201)]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                MEMORY.validate_limits(*arguments)
        self.assertEqual(ctypes.sizeof(MEMORY.RUsageInfoV4), 296)
        self.assertEqual(MEMORY.RUsageInfoV4.ri_phys_footprint.offset, 72)
        self.assertEqual(MEMORY.RUsageInfoV4.ri_resident_size.offset, 64)
        self.assertEqual(MEMORY.RUsageInfoV4.ri_lifetime_max_phys_footprint.offset, 240)

    def test_unreviewed_sdk_header_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in MEMORY.HEADER_HASHES:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("unreviewed SDK")
            with self.assertRaisesRegex(ValueError, "unreviewed SDK"):
                MEMORY.verify_sdk_headers(root)

    def test_hash_stages_only_fixed_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw"
            data = b"sample" * 20_000
            path.write_bytes(data)
            self.assertEqual(MEMORY.file_sha256(path), hashlib.sha256(data).hexdigest())


CLIENT = r'''
import json, os, socket, sys, time
channel = socket.socket(socket.AF_UNIX)
channel.settimeout(2)
channel.connect(os.environ["TRANSFORMERS_BENCH_MEMORY_SOCKET"])
reader = channel.makefile("rb")
def send(action, phase="synthetic_allocation"):
    message = {"schema_version":1,"id":0,"event":action,"phase":phase}
    channel.sendall(json.dumps(message).encode()+b"\n")
    expected = {"ok":True,"id":0,"event":action}
    assert json.loads(reader.readline()) == expected
send("idle")
send("begin")
'''


class SupervisorLifecycleTests(unittest.TestCase):
    """Actual bounded children exercise failure handling without model inference."""

    def run_child(self, code, timeout=3, sampler=None):
        """Create recoverable isolated evidence directories for one short subprocess."""

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            result = MEMORY.supervise([sys.executable, "-c", code], output, timeout_seconds=timeout,
                                      sampler=sampler or FakeSampler(), abi={"test": True})
            self.assertEqual(result["raw_samples"]["sha256"], MEMORY.file_sha256(output / "samples.jsonl"))
            self.assertEqual(json.loads((output / "report.json").read_text())["status"], result["status"])
            return result

    def test_complete_phase_records_identity_and_distinct_lifetime_counter(self):
        result = self.run_child(CLIENT + 'time.sleep(0.015)\nsend("end", "")\n')
        self.assertEqual(result["status"], "passed", result["errors"])
        self.assertEqual(result["identity"]["process_start_abstime"], 7)
        self.assertEqual(result["phases"][0]["observed"]["sampled_maximum_bytes"]["physical_footprint_bytes"], 80)
        self.assertEqual(result["observed_kernel_reported_lifetime_max_physical_footprint_bytes"], 1080)
        self.assertFalse(result["sampled_maxima_are_exact_unsampled_peaks"])
        self.assertFalse(result["timing_comparable_to_uninstrumented_series"])

    def test_child_exit_unfinished_phase_is_not_success(self):
        result = self.run_child(CLIENT + 'sys.exit(3)\n')
        self.assertEqual(result["status"], "failed")
        self.assertIsNotNone(result["unfinished_phase"])
        self.assertTrue(result["errors"])
        self.assertNotEqual(result["child_exit_code"], 0)

    def test_no_markers_and_timeout_fail_closed(self):
        self.assertEqual(self.run_child("pass")["status"], "failed")
        result = self.run_child("import time; time.sleep(5)", timeout=0.05)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any(error.get("supervisor_error") == "TimeoutError" for error in result["errors"]))

    def test_pid_reuse_or_probe_failure_cannot_be_hidden(self):
        class ChangedIdentity:
            def __init__(self):
                self.calls = 0
            def sample(self, pid):
                self.calls += 1
                result = observation(pid)
                result["process_start_abstime"] += self.calls
                return result
        result = self.run_child("import time; time.sleep(5)", sampler=ChangedIdentity())
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("identity changed" in str(error) for error in result["errors"]))
        class FailedProbe:
            def sample(self, pid):
                raise OSError(13, "denied")
        result = self.run_child("import time; time.sleep(5)", sampler=FailedProbe())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["whole_observation_window"]["probe_errors"], 3)
        self.assertIsNone(result["whole_observation_window"]["sampled_maximum_bytes"]["physical_footprint_bytes"])

    def test_existing_output_and_oversized_event_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileExistsError):
            MEMORY.supervise([sys.executable, "-c", "pass"], Path(directory), sampler=FakeSampler())
        code = 'import socket,os,time\ns=socket.socket(socket.AF_UNIX)\ns.connect(os.environ["TRANSFORMERS_BENCH_MEMORY_SOCKET"])\ns.sendall(b"x"*4097)\ntime.sleep(1)'
        result = self.run_child(code)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("oversized" in str(error) for error in result["errors"]))

    @unittest.skipUnless(sys.platform == "darwin", "read-only Darwin ABI smoke")
    def test_real_darwin_child_allocation_has_observable_footprint_delta(self):
        sdk = Path(subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip())
        abi = MEMORY.verify_sdk_headers(sdk)
        self.assertEqual(abi["flavor"], 4)
        code = CLIENT + 'payload=bytearray(8*1024*1024)\nfor offset in range(0,len(payload),4096): payload[offset]=1\ntime.sleep(0.020)\nsend("end", "")\n'
        result = self.run_child(code, sampler=MEMORY.DarwinSampler())
        self.assertEqual(result["status"], "passed", result["errors"])
        self.assertGreaterEqual(result["phases"][0]["sampled_maximum_minus_idle_median_bytes"]["physical_footprint_bytes"], 4*1024*1024)
        self.assertGreaterEqual(result["phases"][0]["idle_baseline"]["successful_samples"], 5)


if __name__ == "__main__":
    unittest.main()
