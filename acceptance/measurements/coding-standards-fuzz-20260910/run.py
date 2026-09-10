"""Run one bounded campaign against the integrator's already-built ASan binary."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("/Users/nover/repos/transformers")
OUTPUT = ROOT / "acceptance/measurements/coding-standards-fuzz-20260910"
WORK = ROOT / "target/standards-fuzz-experiment-interfaces-20260910"
BINARY = ROOT / "fuzz/target/aarch64-apple-darwin/release/experiment_interfaces"


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def source_inventory():
    paths = [ROOT / p for p in ("Cargo.toml", "Cargo.lock", "fuzz/Cargo.toml", "fuzz/Cargo.lock", "fuzz/evidence/run_refresh.py")]
    for relative in ("src", "fuzz/src", "fuzz/fuzz_targets", "fuzz/seeds"):
        for path in (ROOT / relative).rglob("*"):
            if not path.is_file() or path.suffix in {".md", ".pyc"}:
                continue
            # cfg(test) is false in this ASan build; sibling test additions are
            # outside its compilation. Production files are hashed in full.
            if path.name == "tests.rs" or path.name.endswith("_tests.rs") or "tests" in path.relative_to(ROOT).parts:
                continue
            paths.append(path)
    return {p.relative_to(ROOT).as_posix(): identity(p) for p in sorted(set(paths))}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    WORK.mkdir(parents=True, exist_ok=False)
    (WORK / "corpus").mkdir()
    (WORK / "artifacts").mkdir()
    shutil.copytree(ROOT / "fuzz/seeds/experiment_interfaces", WORK / "seeds")
    shutil.copyfile(__file__, OUTPUT / "run.py")
    helper_path = ROOT / "fuzz/evidence/run_refresh.py"
    specification = importlib.util.spec_from_file_location("audited_fuzz_supervisor", helper_path)
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    before = source_inventory()
    binary_before = identity(BINARY)
    seeds_before = helper.inventory(ROOT / "fuzz/seeds/experiment_interfaces")
    copied_seeds_before = helper.inventory(WORK / "seeds")
    save(OUTPUT / "source-inventory-before.json", before)
    for name in ("fuzz-asan-build.log", "fuzz-asan-build.json", "fuzz-storage.json"):
        shutil.copyfile(ROOT / "target/standards-validation" / name, OUTPUT / name)
    record = {
        "schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_scope": "All current production runtime/harness files, including untracked files, both Cargo manifests/locks, all entrypoints/seeds and executed supervisor. cfg(test)-only sibling files are excluded from this native non-test build. External/prebuilt dependency closure is not hermetically attested.",
        "work_directory": str(WORK), "runner": identity(OUTPUT / "run.py"),
        "binary": {"path": str(BINARY), "resolved_path": str(BINARY.resolve()), **binary_before},
        "build_evidence": {name: identity(OUTPUT / name) for name in ("fuzz-asan-build.log", "fuzz-asan-build.json", "fuzz-storage.json")},
        "storage": {"local_free_bytes_before": shutil.disk_usage(ROOT).free, "external_cache_free_bytes_before": shutil.disk_usage(BINARY.parent).free, "cache_reused": True, "no_build_performed_by_campaign": True},
        "sanitizer": "address", "target": "aarch64-apple-darwin", "build_std": False,
        "environment": {"ASAN_OPTIONS": "detect_odr_violation=0", "ASAN_SYMBOLIZER_PATH": "/opt/homebrew/opt/llvm/bin/llvm-symbolizer", "CARGO_NET_OFFLINE": "true", "PYTHONDONTWRITEBYTECODE": "1"},
        "committed_seeds_before": seeds_before, "copied_seeds_before": copied_seeds_before,
        "qualification": "Bounded host/Flex interface campaign only; no accelerator, model support, package, production coverage or exhaustive correctness claim.",
    }
    save(OUTPUT / "manifest.json", record)
    environment = os.environ.copy()
    for name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "LLVM_PROFILE_FILE"):
        if environment.get(name):
            raise RuntimeError(f"Unexpected inherited execution/compiler override: {name}")
    environment.update(record["environment"])
    command = [str(BINARY), str(WORK / "corpus"), str(WORK / "seeds"),
               f"-artifact_prefix={WORK / 'artifacts'}/", "-runs=10000", "-max_len=4096", "-timeout=10", "-rss_limit_mb=2048", "-seed=20260910", "-print_final_stats=1"]
    started = time.monotonic()
    exit_code = 1
    try:
        for label, argv in (
            ("rustc", ["rustup", "run", "nightly", "rustc", "--version", "--verbose"]),
            ("cargo-fuzz", ["/Users/nover/.cargo/bin/cargo-fuzz", "--version"]),
            ("symbolizer", [record["environment"]["ASAN_SYMBOLIZER_PATH"], "--version"]),
        ):
            result = helper.execute(label, argv, ROOT, environment, OUTPUT, 30)
            if result["native_returncode"] != 0 or result["timed_out"]:
                raise RuntimeError(f"Tool identity probe failed: {label}")
        execution = helper.execute("experiment_interfaces", command, ROOT, environment, OUTPUT, 900)
        diagnostics = (OUTPUT / "experiment_interfaces.log").read_text(errors="replace")
        result = helper.campaign_result(execution, diagnostics, helper.inventory(WORK / "artifacts"))
        record["execution"] = execution
        record["campaign"] = result
        record["corpus_after"] = helper.inventory(WORK / "corpus")
        record["copied_seeds_after"] = helper.inventory(WORK / "seeds")
        record["committed_seeds_after"] = helper.inventory(ROOT / "fuzz/seeds/experiment_interfaces")
        after = source_inventory()
        save(OUTPUT / "source-inventory-after.json", after)
        record["source_unchanged"] = before == after
        record["binary_unchanged"] = binary_before == identity(BINARY)
        record["committed_seeds_unchanged"] = seeds_before == record["committed_seeds_after"]
        record["copied_seeds_unchanged"] = copied_seeds_before == record["copied_seeds_after"]
        passed = result["passed"] and all(record[name] for name in ("source_unchanged", "binary_unchanged", "committed_seeds_unchanged", "copied_seeds_unchanged"))
        record["status"] = "passed" if passed else "failed"
        exit_code = 0 if passed else 1
    except BaseException as error:
        record["status"] = "failed"
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        save(OUTPUT / "source-inventory-after.json", source_inventory())
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["exit_code"] = exit_code
        save(OUTPUT / "manifest.json", record)
        print(json.dumps({"status": record["status"], "output": str(OUTPUT), "campaign": record.get("campaign")}, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
