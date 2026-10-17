"""Runs the "quantized" validation group one process at a time (scratch harness, not committed).

Usage: python3 -I run_quantized_gates.py BINARIES_JSON OUT_DIR NAME [NAME...]

BINARIES_JSON maps a build label (A, B, C, D, Dq, E) to a release test binary. Each NAME
selects one planned run (see PLAN). For every run this script:

* waits while a source gate (full_gate.sh) runs, then takes the shared
  heavy-model lock (mkdir; polled every 60 s) and always releases it after
  the run, including on failure;
* refuses to run unless the worktree is clean, and starts the per-run log with
  `== source <sha> dirty=0`, the binary path and SHA-256, backend/dtype,
  every TRANSFORMERS_* variable, the exact command, load, free memory and swap;
* for a create-only conversion, checks free space on the output volume first
  (at least twice the expected artifact size plus 20 GiB) and refuses an
  existing output path;
* captures the test's complete stdout/stderr and the full `/usr/bin/time -l`
  summary, and closes with an `== exit=` line carrying wall time and the
  watchdog figures;
* after a conversion, hashes the published artifact outside the test process
  and compares it with the authenticated artifact the committed record names.

With GATE_EXPORT_DIR and GATE_EXPORT_SHA set, a run executes from that
`git archive` export of the commit instead of the worktree; the export is
checked against the commit's tree (scratch verify_export.py, read-only, with
an absolute path) and its mismatch count is the logged `dirty` figure.
GATE_LOG_SUFFIX is appended to each log name (for alternating A/B rounds).

Watchdog (sampled once per second, the policy of the earlier replay records):
kill on free memory below 10%, swap growth above 8 GiB, or swap-ins above
1 GiB counted only while free memory is below 30%.
"""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = "/Users/nover/repos/transformers-worktrees/tfm-polish-half"
EXPORT_DIR = os.environ.get("GATE_EXPORT_DIR")
EXPORT_SHA = os.environ.get("GATE_EXPORT_SHA")
WT = EXPORT_DIR or REPO
SUFFIX = os.environ.get("GATE_LOG_SUFFIX", "")
VERIFY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_export.py")
LOCK = "/Users/nover/repos/transformers-worktrees/.heavy-model.lock"
PAGE = 16384
GIB = 1 << 30
MUSE = "/Volumes/bag of holding/models/meta-models/Muse-Glimmer-30B-a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
QWEN = "/Volumes/bag of holding/models/Qwen/Qwen3-30B-A3B-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
ACC = "/Volumes/bag of holding/models/transformers-acceptance"
NEW = f"{ACC}/mac-validation-quantized-20261017"

# Authenticated artifacts named by hash in committed records (never written here).
AUTH = {
    "muse-q8": (f"{ACC}/muse-glimmer-30b-q8-a4e59da5/muse-glimmer-30b-a4e59da5-q8-g64-v2.bpk",
                "91cfd35bfe0a389cbc2eed2189c4e33312bdce1a0ff50d8452acf06d00f0e8eb", 28165645337),
    "muse-q4": (f"{ACC}/muse-glimmer-30b-q4-a4e59da5/muse-glimmer-30b-a4e59da5-q4-g64-v2.bpk",
                "05b69f67799b4dc0228b4298e39d466d753fcc5d6484f61c8920c551017daed3", 14911372485),
    "qwen-q8": (f"{ACC}/qwen3-30b-a3b-q8-ad44e777/qwen3-30b-a3b-ad44e777-q8-g64-v2.bpk",
                "7f5230819e7d2170bd85aa3a68ec2c18c303b06a5322cc811f1fbcae64a78f0c", 32110057725),
    "qwen-q4": (f"{ACC}/qwen3-30b-a3b-q4-ad44e777/qwen3-30b-a3b-ad44e777-q4-g64-v2.bpk",
                "70acdcc7e5c6149b888fefab42d42ebf64cb2f2444af2bdadbf573a91a934dd4", 17006255982),
}
# New create-only outputs (same file names, fresh directories beside the replay's).
OUT = {
    "muse-q8": f"{NEW}/muse-glimmer-30b-q8-a4e59da5/muse-glimmer-30b-a4e59da5-q8-g64-v2.bpk",
    "muse-q4": f"{NEW}/muse-glimmer-30b-q4-a4e59da5/muse-glimmer-30b-a4e59da5-q4-g64-v2.bpk",
    "qwen-q8": f"{NEW}/qwen3-30b-a3b-q8-ad44e777/qwen3-30b-a3b-ad44e777-q8-g64-v2.bpk",
    "qwen-q4": f"{NEW}/qwen3-30b-a3b-q4-ad44e777/qwen3-30b-a3b-ad44e777-q4-g64-v2.bpk",
}


def burnpack(key):
    """Returns the Burnpack a smoke loads: the new output, or the authenticated one if forced."""
    return AUTH[key][0] if os.environ.get("GATE_USE_AUTHENTICATED") == "1" else OUT[key]


def plan():
    """Returns {name: (binary label, backend/dtype label, exact test, env, conversion key or None)}."""
    muse = {"TRANSFORMERS_MUSE_GLIMMER_DIR": MUSE}
    qwen = {"TRANSFORMERS_QWEN3_MOE_DIR": QWEN}
    p = {
        "packed-metal-kernels": ("A", "Metal/packed Q4,Q8 (F16 and F32 activations)",
                                 "packed_q4_q8_matmul_launches_quantized_metal_kernels", {}, None),
        "lib-metal-q4-multirow": ("B", "Metal/packed Q4 G64, F32 activations, 1-64 rows",
                                  "operations::tests::metal_qfloat_projection_executes_quantized_matmul_dispatch",
                                  {}, None),
        "lib-metal-q4-qwen35-shapes": ("B", "Metal/packed Q4, F16 activations, 1/3/17 rows",
                                       "operations::tests::metal_qfloat_projection_covers_qwen35_shapes",
                                       {}, None),
        "lib-metal-2bit-rejected": ("B", "Metal/packed Q2 rejection",
                                    "operations::tests::metal_two_bit_packed_projection_is_rejected",
                                    {}, None),
        "lib-metal-packed-transpose": ("B", "Metal/packed install transpose",
                                       "checkpoint::runtime::packed_install::tests::metal_quantized_transpose_is_exact_and_stays_packed",
                                       {}, None),
        "muse-q8-conversion": ("C", "host conversion BF16->F16->Q8/G64",
                               "quantizes_pinned_muse_glimmer_weights_to_q8_burnpack",
                               {**muse, "TRANSFORMERS_MUSE_GLIMMER_Q8_BURNPACK": OUT["muse-q8"]}, "muse-q8"),
        "muse-q4-conversion": ("C", "host conversion BF16->F16->Q4/G64",
                               "quantizes_pinned_muse_glimmer_weights_to_q4_burnpack",
                               {**muse, "TRANSFORMERS_MUSE_GLIMMER_Q4_BURNPACK": OUT["muse-q4"]}, "muse-q4"),
        "muse-q8-smoke": ("D", "Metal/F16 activations + packed Q8/G64 (kernel-traced)",
                          "smokes_and_audits_pinned_muse_glimmer_q8_burnpack_on_metal",
                          {**muse, "TRANSFORMERS_MUSE_GLIMMER_Q8_BURNPACK": burnpack("muse-q8")}, None),
        "muse-q4-smoke": ("D", "Metal/F16 activations + packed Q4/G64 (kernel-traced)",
                          "smokes_and_audits_pinned_muse_glimmer_q4_burnpack_on_metal",
                          {**muse, "TRANSFORMERS_MUSE_GLIMMER_Q4_BURNPACK": burnpack("muse-q4")}, None),
        "qwen-q8-conversion": ("E", "host conversion BF16->F16->Q8/G64",
                               "quantizes_pinned_qwen3_moe_weights_to_q8_burnpack",
                               {**qwen, "TRANSFORMERS_QWEN3_MOE_Q8_BURNPACK": OUT["qwen-q8"]}, "qwen-q8"),
        "qwen-q4-conversion": ("E", "host conversion BF16->F16->Q4/G64",
                               "quantizes_pinned_qwen3_moe_weights_to_q4_burnpack",
                               {**qwen, "TRANSFORMERS_QWEN3_MOE_Q4_BURNPACK": OUT["qwen-q4"]}, "qwen-q4"),
        "qwen-q8-smoke": ("E", "Metal/F16 activations + packed Q8/G64",
                          "loads_pinned_qwen3_moe_q8_and_runs_metal_smoke",
                          {**qwen, "TRANSFORMERS_QWEN3_MOE_Q8_BURNPACK": burnpack("qwen-q8")}, None),
        "qwen-q4-smoke": ("E", "Metal/F16 activations + packed Q4/G64",
                          "loads_pinned_qwen3_moe_q4_and_runs_metal_smoke",
                          {**qwen, "TRANSFORMERS_QWEN3_MOE_Q4_BURNPACK": burnpack("qwen-q4")}, None),
        "qwen-q8-audit": ("Dq", "Metal/F16 activations + packed Q8/G64 (kernel-traced)",
                          "audits_pinned_qwen3_moe_q8_decode_packed_metal_kernels",
                          {**qwen, "TRANSFORMERS_QWEN3_MOE_Q8_BURNPACK": burnpack("qwen-q8")}, None),
        "qwen-q4-audit": ("Dq", "Metal/F16 activations + packed Q4/G64 (kernel-traced)",
                          "audits_pinned_qwen3_moe_q4_decode_packed_metal_kernels",
                          {**qwen, "TRANSFORMERS_QWEN3_MOE_Q4_BURNPACK": burnpack("qwen-q4")}, None),
    }
    return p


def free_pct():
    """Returns memory_pressure's system-wide free percentage."""
    out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    return int(re.search(r"free percentage: (\d+)%", out).group(1))


def swap_used_mib():
    """Returns used swap in MiB."""
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1))


def swapins():
    """Returns the cumulative swap-in page count."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int(re.search(r"Swapins:\s+(\d+)", out).group(1))


def source():
    """Returns (commit sha, mismatch count) of the export, or (HEAD sha, porcelain lines) of the worktree."""
    if EXPORT_DIR:
        out = subprocess.run([sys.executable, "-I", VERIFY, REPO, EXPORT_SHA, os.path.abspath(EXPORT_DIR)],
                             capture_output=True, text=True, check=True).stdout
        return EXPORT_SHA, int(re.search(r"dirty=(\d+)", out).group(1))
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True,
                           text=True, check=True).stdout
    return sha, len(dirty.splitlines())


def sha256_file(path):
    """Streams one file through SHA-256."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    """Returns the current UTC time in ISO form."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def acquire_lock(summary):
    """Waits for source gates, then takes the heavy-model lock (polling every 60 s)."""
    started = time.monotonic()
    while True:
        while subprocess.run(["pgrep", "-f", "[f]ull_gate.sh"], capture_output=True).returncode == 0:
            time.sleep(60)
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            time.sleep(60)
    waited = time.monotonic() - started
    summary.write(f"{now()} lock acquired (waited {waited:.0f}s)\n")
    return waited


def run_one(name, spec, binaries, out_dir, summary):
    """Runs one planned test under the lock and writes its complete log."""
    label_key, label, test, extra, conversion = spec
    binary = binaries[label_key]
    waited = acquire_lock(summary)
    try:
        digest = hashlib.sha256(open(binary, "rb").read()).hexdigest()
        sha, dirty = source()
        if dirty:
            summary.write(f"== refusing {name}: dirty tree ({dirty} lines)\n")
            return 3
        path = os.path.join(out_dir, name + SUFFIX + ".log")
        with open(path, "w", buffering=1) as log:
            log.write(f"== source {sha} dirty={dirty}\n")
            if EXPORT_DIR:
                log.write(f"== export {EXPORT_DIR} (git archive of {sha}, verified against its tree)\n")
            log.write(f"== binary {binary} sha256 {digest} (build {label_key})\n")
            log.write(f"== {label} {test}\n")
            for key in sorted(extra):
                log.write(f"== env {key}={extra[key]}\n")
            if conversion:
                output = OUT[conversion]
                parent = os.path.dirname(output)
                os.makedirs(parent, exist_ok=True)
                free = shutil.disk_usage(parent).free
                need = 2 * AUTH[conversion][2] + 20 * GIB
                log.write(f"== preflight output_parent={parent} free_bytes={free} "
                          f"required_bytes={need} output_exists={os.path.lexists(output)}\n")
                if free < need or os.path.lexists(output):
                    log.write("== refusing: insufficient space or existing output\n")
                    summary.write(f"== refusing {name}: preflight failed\n")
                    return 4
            cmd = ["/usr/bin/time", "-l", binary, "--ignored", "--exact", "--nocapture",
                   "--test-threads", "1", test]
            env = {k: v for k, v in os.environ.items() if not k.startswith("TRANSFORMERS_")}
            env.update(extra)
            log.write(f"== cwd {WT}\n== cmd {' '.join(cmd)}\n")
            load = os.getloadavg()
            log.write(f"== start {now()} lock_wait_s {waited:.0f} load {load[0]:.2f} {load[1]:.2f} "
                      f"{load[2]:.2f} free_pct {free_pct()} swap_used_mib {swap_used_mib():.0f}\n")
            log.flush()
            swap0, swapin0 = swap_used_mib(), swapins()
            last_swapin, pressured = swapin0, 0
            min_free, max_growth, reason = 100, 0.0, None
            start = time.monotonic()
            proc = subprocess.Popen(cmd, cwd=WT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while proc.poll() is None:
                time.sleep(1)
                free = free_pct()
                growth = swap_used_mib() - swap0
                current = swapins()
                if free < 30:
                    pressured += current - last_swapin
                last_swapin = current
                min_free, max_growth = min(min_free, free), max(max_growth, growth)
                if free < 10:
                    reason = f"free memory {free}%"
                elif growth > 8192:
                    reason = f"swap grew by {growth:.0f} MiB"
                elif pressured * PAGE > GIB:
                    reason = f"pressured swap-ins {pressured * PAGE >> 20} MiB"
                if reason:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    break
            wall = time.monotonic() - start
            total_swapin = (swapins() - swapin0) * PAGE >> 20
            load = os.getloadavg()
            line = (f"== exit={proc.returncode} wall={wall:.1f}s min_free_pct={min_free} "
                    f"max_swap_growth_mib={max_growth:.0f} swapin_mib={total_swapin} "
                    f"pressured_swapin_mib={pressured * PAGE >> 20} end={now()} "
                    f"end_load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}"
                    + (f" watchdog=\"{reason}\"" if reason else ""))
            log.write(line + "\n")
            if conversion and proc.returncode == 0:
                output = OUT[conversion]
                started = time.monotonic()
                observed = sha256_file(output)
                size = os.path.getsize(output)
                auth_path, auth_sha, auth_bytes = AUTH[conversion]
                log.write(f"== artifact {output} bytes={size} sha256={observed} "
                          f"(hashed outside the test in {time.monotonic() - started:.1f}s)\n")
                log.write(f"== authenticated {auth_path} bytes={auth_bytes} sha256={auth_sha} "
                          f"identical={observed == auth_sha and size == auth_bytes}\n")
        summary.write(f"{now()} {name}{SUFFIX} {label} {test}: {line}\n")
        return proc.returncode
    finally:
        os.rmdir(LOCK)
        summary.write(f"{now()} lock released\n")


def main():
    """Runs each named plan entry in order, stopping at the first refusal."""
    binaries = json.load(open(sys.argv[1]))
    out_dir = sys.argv[2]
    names = sys.argv[3:]
    p = plan()
    summary = open(os.path.join(out_dir, "summary.log"), "a", buffering=1)
    for name in names:
        code = run_one(name, p[name], binaries, out_dir, summary)
        if code in (3, 4):
            sys.exit(code)


if __name__ == "__main__":
    main()
