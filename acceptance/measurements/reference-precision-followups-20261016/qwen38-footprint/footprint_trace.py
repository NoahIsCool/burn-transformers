#!/usr/bin/env python3
"""Run a command and sample its physical footprint (proc_pid_rusage, RUSAGE_INFO_V4) every 50 ms.

Usage: footprint_trace.py OUTPUT.tsv -- COMMAND...
Writes `seconds<TAB>footprint_bytes` rows and prints the peak and when it occurred.
"""

import ctypes
import subprocess
import sys
import time

libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
RUSAGE_INFO_V4 = 4


class RusageInfoV4(ctypes.Structure):
    """Leading fields of struct rusage_info_v4 up to ri_phys_footprint (Darwin libproc.h)."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_rest", ctypes.c_uint64 * 64),
    ]


def footprint(pid: int) -> int | None:
    """Current physical footprint of `pid`, or None once it has exited."""

    info = RusageInfoV4()
    if libproc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return info.ri_phys_footprint


def main() -> int:
    output = sys.argv[1]
    command = sys.argv[3:] if sys.argv[2] == "--" else sys.argv[2:]
    started = time.monotonic()
    child = subprocess.Popen(command)
    rows = []
    while child.poll() is None:
        value = footprint(child.pid)
        if value is not None:
            rows.append((time.monotonic() - started, value))
        time.sleep(0.05)
    with open(output, "w") as handle:
        for seconds, value in rows:
            handle.write(f"{seconds:.3f}\t{value}\n")
    peak_seconds, peak = max(rows, key=lambda row: row[1])
    end = rows[-1][0]
    print(f"peak={peak} at {peak_seconds:.2f}s of {end:.2f}s samples={len(rows)} exit={child.returncode}")
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
