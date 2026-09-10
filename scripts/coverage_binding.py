#!/usr/bin/env python3
"""Bind one fresh coverage run to native compiler inputs and LLVM export objects.

This opt-in development tool supports the pinned nightly producer in
coverage_dependencies.py and cargo-llvm-cov 0.6.21. Its published source package
SHA256 is 42236fed339535379c8671218bd8050ddf2b937cb16f6012b44b6cf993ce7c9c
(VCS 31b63e9d6b24bc337e8b138edd6514fe903b7889). context.rs selects LLVM_COV
and LLVM_PROFDATA together; main.rs::Format::generate_report supplies explicit
objects. clean.rs::clean_partial cleans measured workspace artifacts by default.
The installed binary's override and Cargo wrapper cache behavior are exercised
by the optional native tests, not inferred solely from version strings.

Cargo rewrites native JSON through its own serde schema. We retain the exact
native stdout separately; only that report is directly bound to native objects.
No counters or exclusions are adopted here. Compiler-reported input closure is
scoped to the selected workspace targets/configuration, NOT arbitrary untracked
procedural-macro I/O, hostile edit-and-restore races, or inline macro spans.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from coverage_compiler import MAX_ARTIFACT_BYTES, unique_object, reject_constant
from coverage_dependencies import COVERAGE_PRODUCER, parse_dep_info, require_supported_producer
from coverage_provenance import fingerprint_file


MAX_RECORD_BYTES = 8 * 1024 * 1024
MAX_RECORDS = 512
MAX_EXPORT_BYTES = 2 * 1024 * 1024 * 1024
SCRIPT = Path(__file__).resolve()
COMPILER_SCRIPT = SCRIPT.with_name("coverage_compiler.py")


def read_json(path: Path, maximum: int = MAX_RECORD_BYTES) -> dict:
    """Read one bounded, non-symlink JSON object with unambiguous keys."""

    identity = fingerprint_file(path, maximum_bytes=maximum)
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum or fingerprint_file(path, maximum_bytes=maximum) != identity:
        raise ValueError("coverage evidence changed during read")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("invalid coverage evidence JSON") from error
    if not isinstance(value, dict):
        raise ValueError("coverage evidence must be a JSON object")
    return value


def write_json(path: Path, value: dict) -> None:
    """Exclusively retain an observation without replacing earlier evidence."""

    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def contained_file(value: str, root: Path) -> Path:
    """Require an absolute non-symlink file below its explicit evidence root."""

    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("invalid evidence path")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("evidence path must be absolute and not a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("evidence path escapes its configured root")
    return resolved


def require_identity(path: Path, identity: dict, maximum: int = MAX_ARTIFACT_BYTES) -> None:
    """Reject stale, replaced, oversized or otherwise changed recorded bytes."""

    if identity != fingerprint_file(path, maximum_bytes=maximum):
        raise ValueError("recorded coverage artifact identity changed")


def producer_from_version(version: str):
    """Parse exact rustc -Vv fields, rejecting duplicate or unreviewed producers."""

    if not isinstance(version, str) or not version.startswith("rustc "):
        raise ValueError("missing native rustc identity")
    fields = {}
    for line in version.splitlines()[1:]:
        key, separator, value = line.partition(": ")
        if not separator or key in fields:
            raise ValueError("ambiguous native rustc identity")
        fields[key] = value
    producer = require_supported_producer(release=fields["release"], commit_hash=fields["commit-hash"],
                                          llvm_version=fields["LLVM version"])
    if producer != COVERAGE_PRODUCER:
        raise ValueError("compiler evidence coverage requires the pinned LLVM 21.1.8 nightly")
    return producer


def export_inputs(arguments: list[str]) -> tuple[str, list[str]]:
    """Accept only the exact full-JSON export shape used by the pinned runner."""

    if arguments[:2] != ["export", "-format=text"] or len(arguments) > 2 * MAX_RECORDS + 8:
        raise ValueError("unsupported LLVM export command")
    profile = None
    objects = []
    index = 2
    while index < len(arguments):
        argument = arguments[index]
        index += 1
        if argument.startswith("-instr-profile=") and profile is None:
            profile = argument.partition("=")[2]
        elif argument == "-object" and index < len(arguments):
            value = arguments[index]
            index += 1
            if not value or value.startswith("@"):
                raise ValueError("invalid LLVM export option value")
            objects.append(value)
        else:
            raise ValueError("unsupported or duplicate LLVM export option")
    if not profile or not objects or len(set(objects)) != len(objects):
        raise ValueError("LLVM export requires one profile and distinct explicit objects")
    return profile, objects


def capture_export(tool: Path, root: Path, artifact_root: Path, arguments: list[str]) -> int:
    """Retain exact native stdout and pre/post input hashes before forwarding it.

    Native stderr flows directly into the runner's diagnostic log. Like rustc,
    the selected local LLVM executable is trusted; output is streamed with a
    size cap, and its child is killed/drained on a forwarding or read failure.
    A failed record never qualifies input binding even if LLVM returned zero.
    """

    record = {"schema_version": 1, "command": [str(tool), *arguments], "cwd": str(Path.cwd()),
              "capture_valid": False, "process_exit_code": None}
    try:
        profile, objects = export_inputs(arguments)
        # cargo-llvm-cov shortens object filenames relative to its actual cwd.
        # Preserve the literal argv, then resolve those paths against that cwd.
        paths = [contained_file(str(Path.cwd() / name), artifact_root) for name in [profile, *objects]]
        if len(set(paths)) != len(paths):
            raise ValueError("aliased LLVM export input paths")
        record["tool_identity"] = fingerprint_file(tool, maximum_bytes=MAX_ARTIFACT_BYTES)
        record["inputs"] = [{"path": str(path), "identity": fingerprint_file(path, maximum_bytes=MAX_ARTIFACT_BYTES)}
                            for path in paths]
        report_path = root / "native-coverage.json"
        with report_path.open("xb") as output:
            with subprocess.Popen([str(tool), *arguments], stdout=subprocess.PIPE) as child:
                try:
                    assert child.stdout is not None
                    total = 0
                    while chunk := child.stdout.read(64 * 1024):
                        total += len(chunk)
                        if total > MAX_EXPORT_BYTES:
                            raise ValueError("native LLVM export exceeds byte bound")
                        output.write(chunk)
                    record["process_exit_code"] = child.wait()
                except BaseException:
                    child.kill()
                    child.wait()
                    raise
        record["report"] = {"path": str(report_path), "identity": fingerprint_file(report_path, maximum_bytes=MAX_EXPORT_BYTES)}
        if record["process_exit_code"] == 0:
            for entry in record["inputs"]:
                require_identity(Path(entry["path"]), entry["identity"])
            require_identity(tool, record["tool_identity"])
            record["capture_valid"] = True
        write_json(root / "export.json", record)
        with report_path.open("rb") as output:
            while chunk := output.read(64 * 1024):
                sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        return record["process_exit_code"]
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        record["capture_valid"] = False
        record["capture_error"] = str(error)
        if not (root / "export.json").exists():
            write_json(root / "export.json", record)
        print(f"error: native LLVM evidence capture: {error}", file=sys.stderr)
        return 1


def launcher(path: Path, script: Path, arguments: list[str]) -> dict:
    """Create an exact run-local executable; its unique path invalidates Cargo cache."""

    with path.open("x", encoding="utf-8") as stream:
        stream.write("#!/bin/sh\nexec " + shlex.join([sys.executable, str(script), *arguments]) + ' "$@"\n')
    path.chmod(0o700)
    return fingerprint_file(path)


def query(command: list[str], crate: Path, environment: dict) -> str:
    """Read bounded tool metadata, rejecting any stderr or unsuccessful status.

    The tools are trusted local executables. Unlinked temporary streams bound
    retained memory; a timeout limits duration, not a hostile tool's disk writes.
    This deliberately treats even a successful query with warnings as invalid.
    """

    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        result = subprocess.run(command, cwd=crate, env=environment, stdout=output, stderr=errors,
                                check=False, timeout=30)
        output.seek(0)
        errors.seek(0)
        payload = output.read(MAX_RECORD_BYTES + 1)
        if result.returncode != 0 or errors.read(1) or len(payload) > MAX_RECORD_BYTES:
            raise ValueError("coverage metadata query failed, exceeded bounds or emitted diagnostics")
    try:
        return payload.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("coverage metadata query is not UTF-8") from error


def expected_targets(metadata: dict, crate: Path, features: set[str]) -> list[dict]:
    """Restrict capture to one standalone workspace package's selected test targets."""

    packages = metadata["packages"]
    selected = [package for package in packages if Path(package["manifest_path"]).resolve() == crate / "Cargo.toml"]
    if len(selected) != 1 or metadata["workspace_members"] != [selected[0]["id"]]:
        raise ValueError("compiler evidence requires one standalone workspace package")
    result = []
    for target in selected[0]["targets"]:
        if not set(target.get("required-features", [])).issubset(features):
            continue
        kind = target["kind"]
        if kind not in (["lib"], ["test"]):
            if kind == ["custom-build"]:
                raise ValueError("workspace build scripts need additional input evidence")
            continue
        source = Path(target["src_path"]).resolve(strict=True)
        if not source.is_relative_to(crate):
            raise ValueError("workspace target source escapes crate")
        entry = {"name": target["name"].replace("-", "_"), "source": str(source), "test": True}
        result.append(entry)
        if kind == ["lib"]:
            result.append({**entry, "test": False})
    if sum(not item["test"] for item in result) != 1 or len(result) < 3:
        raise ValueError("compiler evidence requires production, unit and integration targets")
    if len({(item["name"], item["source"], item["test"]) for item in result}) != len(result):
        raise ValueError("workspace target identities cannot be distinguished in compiler argv")
    return result


def prepare_capture(crate: Path, output: Path, toolchain: str, features: str, environment: dict) -> tuple[dict, dict]:
    """Configure fresh workspace and export wrappers without touching cached dependencies."""

    forbidden = ("RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
                 "CARGO_TARGET_DIR", "CARGO_BUILD_TARGET", "CARGO_BUILD_BUILD_DIR", "CARGO_BUILD_RUSTC",
                 "LLVM_COV", "LLVM_PROFDATA", "LLVM_COV_FLAGS", "LLVM_PROFDATA_FLAGS",
                 "CARGO_LLVM_COV_FLAGS", "CARGO_LLVM_PROFDATA_FLAGS", "CARGO_LLVM_COV_TARGET_DIR",
                 "CARGO_LLVM_COV_BUILD_DIR", "CARGO_LLVM_COV_SHOW_ENV", "LLVM_PROFILE_FILE", "LLVM_PROFILE_FILE_NAME")
    if any(environment.get(name) for name in forbidden):
        raise ValueError("compiler evidence rejects inherited compiler/exporter overrides")
    invocation = ["cargo", f"+{toolchain}"]
    version = query(["rustc", f"+{toolchain}", "-Vv"], crate, environment)
    producer_from_version(version)
    exporter_version = query(invocation + ["llvm-cov", "--version"], crate, environment)
    if exporter_version.strip() != "cargo-llvm-cov 0.6.21":
        raise ValueError("unsupported cargo-llvm-cov producer")
    target_lib = query(["rustc", f"+{toolchain}", "--print", "target-libdir"], crate, environment).strip()
    tools = Path(target_lib).parent / "bin"
    llvm_cov, llvm_profdata = tools / "llvm-cov", tools / "llvm-profdata"
    rustc = Path(query(["rustup", "which", "--toolchain", toolchain, "rustc"], crate, environment).strip()).resolve(strict=True)
    tool_identities = {str(path): fingerprint_file(path, maximum_bytes=MAX_ARTIFACT_BYTES) for path in (llvm_cov, llvm_profdata, rustc)}
    llvm_version = query([str(llvm_cov), "--version"], crate, environment)
    if "  LLVM version 21.1.8-rust-1.94.0-nightly\n" not in llvm_version:
        raise ValueError("unsupported native LLVM exporter identity")
    metadata_command = invocation + ["metadata", "--manifest-path", str(crate / "Cargo.toml"), "--locked", "--offline",
                                     "--no-deps", "--format-version", "1", "--no-default-features", "--features", features]
    metadata = json.loads(query(metadata_command, crate, environment), object_pairs_hook=unique_object, parse_constant=reject_constant)
    if Path(metadata["target_directory"]).resolve() != crate / "target":
        raise ValueError("compiler evidence requires the crate's normal target directory")
    targets = expected_targets(metadata, crate, set(features.split(",")))
    artifact_root = crate / "target/llvm-cov-target"
    artifact_root.mkdir(exist_ok=True)
    compiler_root, export_root = output / "compiler", output / "llvm-export"
    compiler_root.mkdir()
    export_root.mkdir()
    compiler_wrapper, export_wrapper = output / "rustc-workspace-wrapper", output / "llvm-cov-wrapper"
    wrappers = {str(compiler_wrapper): launcher(compiler_wrapper, COMPILER_SCRIPT, []),
                str(export_wrapper): launcher(export_wrapper, SCRIPT, ["export", str(llvm_cov), str(export_root), str(artifact_root)])}
    configured = {**environment, "RUSTC_WRAPPER": "", "RUSTC_WORKSPACE_WRAPPER": str(compiler_wrapper),
                  "LLVM_COV": str(export_wrapper), "LLVM_PROFDATA": str(llvm_profdata),
                  "TFM_COVERAGE_CAPTURE_ROOT": str(compiler_root), "TFM_COVERAGE_ARTIFACT_ROOT": str(artifact_root),
                  "CARGO_LLVM_COV_SETUP": "no", "CARGO_LLVM_COV_DENY_WARNINGS": "1"}
    setup = {"schema_version": 1, "compiler_root": str(compiler_root), "export_root": str(export_root),
             "artifact_root": str(artifact_root), "targets": targets, "wrappers": wrappers,
             "tools": tool_identities, "compiler_version": version, "cargo_llvm_cov_version": exporter_version,
             "llvm_version": llvm_version, "rustc": str(rustc), "llvm_cov": str(llvm_cov), "metadata_command": metadata_command}
    return configured, setup


def compiler_arguments(command: list[str], cwd: Path) -> dict:
    """Parse Cargo's bounded literal rustc argv; unsupported options fail closed."""

    if not isinstance(command, list) or not 2 <= len(command) <= 4096 or any(not isinstance(arg, str) or "\0" in arg or arg.startswith("@") for arg in command):
        raise ValueError("invalid compiler argument vector")
    paired = {"--crate-name", "--edition", "--error-format", "--json", "--crate-type", "--emit", "-C", "--cfg",
              "--check-cfg", "--out-dir", "-L", "--extern", "--cap-lints", "--target", "--diagnostic-width", "--color"}
    options = {}
    sources = []
    test = False
    index = 1
    while index < len(command):
        argument = command[index]
        index += 1
        if argument == "--test" and not test:
            test = True
        elif argument.partition("=")[0] in paired:
            key, separator, value = argument.partition("=")
            if not separator:
                if index >= len(command):
                    raise ValueError("missing compiler option value")
                value = command[index]
                index += 1
            options.setdefault(key, []).append(value)
        elif argument.startswith("-"):
            raise ValueError("unsupported compiler option for source binding")
        else:
            sources.append(str((cwd / argument).resolve(strict=True)))
    if len(sources) != 1 or len(options.get("--crate-name", [])) != 1:
        raise ValueError("compiler invocation has ambiguous source or crate name")
    instrumentation = [value for value in options.get("-C", []) if value.startswith("instrument-coverage")]
    if instrumentation != ["instrument-coverage"]:
        raise ValueError("compiler invocation lacks exact coverage instrumentation")
    return {"name": options["--crate-name"][0], "source": sources[0], "test": test,
            "cfg": sorted(options.get("--cfg", [])), "extern": options.get("--extern", [])}


def validate_capture(setup: dict, snapshot: dict, crate: Path) -> dict:
    """Bind every selected target, compiler-reported input and exact export object.

    All compiler-reported workspace inputs must be in the unchanged source
    inventory. Every LLVM object must be a freshly captured link artifact; the
    production rlib must be linked by each integration target. This is stronger
    than a filename match but is not proof of arbitrary macro filesystem I/O.
    """

    for path, identity in {**setup["wrappers"], **setup["tools"]}.items():
        require_identity(Path(path), identity)
    root, artifact_root = Path(setup["compiler_root"]), Path(setup["artifact_root"])
    entries = []
    with os.scandir(root) as directory:
        for entry in directory:
            entries.append(entry.name)
            if len(entries) > MAX_RECORDS * 2:
                raise ValueError("too many compiler evidence records")
    record_names = sorted(name for name in entries if name.startswith("rustc-") and name.endswith(".json"))
    if not record_names:
        raise ValueError("no fresh compiler captures; cache hits cannot qualify")
    observed_targets, links, dependencies, records, retained = [], {}, [], [], set(record_names)
    for name in record_names:
        record = read_json(root / name)
        if type(record.get("schema_version")) is not int or record["schema_version"] != 1 or record.get("capture_valid") is not True or type(record.get("process_exit_code")) is not int or record["process_exit_code"] != 0:
            raise ValueError("failed or unsupported compiler capture")
        if record.get("compiler_version") != setup["compiler_version"] or Path(record["cwd"]).resolve() != crate:
            raise ValueError("compiler producer or working directory mismatch")
        producer_from_version(record["compiler_version"])
        target = compiler_arguments(record["command"], crate)
        if Path(record["command"][0]).resolve(strict=True) != Path(setup["rustc"]):
            raise ValueError("compiler executable differs from configured native rustc")
        key = {field: target[field] for field in ("name", "source", "test")}
        if key not in setup["targets"] or key in observed_targets:
            raise ValueError("unexpected or duplicate compiled workspace target")
        observed_targets.append(key)
        artifacts = record["artifacts"]
        if not isinstance(artifacts, list) or not 2 <= len(artifacts) <= 16 or sum(a["emit"] == "dep-info" for a in artifacts) != 1:
            raise ValueError("incomplete compiler artifact inventory")
        target_links = []
        for artifact in artifacts:
            path = contained_file(artifact["path"], artifact_root)
            require_identity(path, artifact["identity"])
            if artifact["emit"] == "dep-info":
                copy_name = artifact["retained_dep_info"]
                if copy_name != name.removesuffix(".json") + ".d":
                    raise ValueError("retained dependency filename mismatch")
                retained.add(copy_name)
                copy = root / copy_name
                require_identity(copy, artifact["identity"], MAX_RECORD_BYTES)
                parsed = parse_dep_info(copy.read_text(encoding="utf-8"))
                inputs = []
                for dependency in parsed.dependencies:
                    source = (crate / dependency).resolve(strict=True)
                    if not source.is_relative_to(crate) or source.relative_to(crate).as_posix() not in snapshot["files"]:
                        raise ValueError("compiler input is outside captured source inventory")
                    relative = source.relative_to(crate).as_posix()
                    require_identity(source, snapshot["files"][relative])
                    inputs.append(relative)
                if Path(target["source"]).relative_to(crate).as_posix() not in inputs:
                    raise ValueError("dep-info omits compiler root source")
                dependencies.append({"target": key, "inputs": sorted(set(inputs)),
                                     "environment_names": [item.name for item in parsed.environment]})
            elif artifact["emit"] == "link":
                if str(path) in links:
                    raise ValueError("duplicate compiler link artifact")
                links[str(path)] = artifact["identity"]
                target_links.append(str(path))
            elif artifact["emit"] != "metadata":
                raise ValueError("unsupported compiler artifact kind")
        if len(target_links) != 1:
            raise ValueError("selected workspace target needs exactly one link artifact")
        records.append({**target, "link": target_links[0]})
    if set(entries) != retained or len(observed_targets) != len(setup["targets"]):
        raise ValueError("incomplete or extra compiler evidence inventory")
    production = next(item for item in records if not item["test"])
    for target in records:
        if target["cfg"] != production["cfg"]:
            raise ValueError("workspace target cfg differs from production library")
        if (target["name"], target["source"]) != (production["name"], production["source"]):
            matches = [value.partition("=")[2] for value in target["extern"] if value.partition("=")[0] == production["name"]]
            if len(matches) != 1 or str(Path(matches[0]).resolve(strict=True)) != production["link"]:
                raise ValueError("integration target is not linked to captured production library")
    export_root = Path(setup["export_root"])
    export = read_json(export_root / "export.json")
    if type(export.get("schema_version")) is not int or export["schema_version"] != 1 or export.get("capture_valid") is not True or type(export.get("process_exit_code")) is not int or export["process_exit_code"] != 0:
        raise ValueError("missing or invalid native LLVM export")
    if Path(export["cwd"]).resolve() != crate:
        raise ValueError("native LLVM working directory differs from compiler cwd")
    if not isinstance(export.get("command"), list) or not export["command"] or any(not isinstance(arg, str) for arg in export["command"]):
        raise ValueError("invalid native LLVM command vector")
    tool = export["command"][0]
    if tool != setup["llvm_cov"] or export["tool_identity"] != setup["tools"][tool]:
        raise ValueError("native LLVM executable is not the configured tool")
    profile, objects = export_inputs(export["command"][1:])
    expected_inputs = [str(contained_file(str(crate / value), artifact_root)) for value in [profile, *objects]]
    if [entry["path"] for entry in export["inputs"]] != expected_inputs:
        raise ValueError("native LLVM input inventory differs from argv")
    for entry in export["inputs"]:
        require_identity(Path(entry["path"]), entry["identity"])
    test_links = {item["link"] for item in records if item["test"]}
    if set(expected_inputs[1:]) != test_links:
        raise ValueError("LLVM objects do not exactly match fresh workspace test artifacts")
    for entry in export["inputs"][1:]:
        if entry["identity"] != links[entry["path"]]:
            raise ValueError("LLVM object differs from compiler artifact")
    report = contained_file(export["report"]["path"], export_root)
    if report != export_root / "native-coverage.json":
        raise ValueError("unexpected native report path")
    require_identity(report, export["report"]["identity"], MAX_EXPORT_BYTES)
    native = read_json(report, MAX_EXPORT_BYTES)
    if native.get("type") != "llvm.coverage.json.export" or native.get("version") != "3.0.1":
        raise ValueError("unsupported native LLVM export schema")
    all_inputs = {path for target in dependencies for path in target["inputs"]}
    for data in native["data"]:
        for entry in data["files"]:
            path = (crate / entry["filename"]).resolve()
            if path.is_relative_to(crate / "src") and path.relative_to(crate).as_posix() not in all_inputs:
                raise ValueError("LLVM source is absent from compiler-reported inputs")
    return {"schema_version": 1, "artifact_binding_verified": True, "compiler_reported_inputs_verified": True,
            "no_export_filter_verified": True,
            "compiler_dependency_closure_verified": False, "production_only_exclusions_qualified": False,
            "scope": "compiler-reported workspace inputs for the captured targets and cfg; no arbitrary macro I/O proof",
            "native_report": export["report"], "targets": dependencies,
            "production_link": production["link"], "llvm_objects": sorted(test_links)}


if __name__ == "__main__":
    if len(sys.argv) < 7 or sys.argv[1] != "export":
        raise SystemExit("coverage_binding is an internal explicit-export wrapper")
    raise SystemExit(capture_export(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:]))
