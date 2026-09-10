"""Qualify only whole test-source files for an explicitly non-promoting diagnostic.

The intersection is deliberately narrow: a whole-file syntax candidate must be
an authenticated unit-test input and absent from every captured production input
set. Unknown, mixed, inline and generated ownership stays in the denominator.
Compiler-reported inputs do not prove arbitrary untracked procedural-macro I/O.
Retained inline tests can inflate coverage, so this diagnostic is NOT a lower
bound on production coverage and cannot qualify the production-only release gate.

``qualify_capture`` is the evidence-bearing entry point: it revalidates the
existing native binding and a fresh classifier invocation before joining them.
``whole_file_diagnostic`` is a pure measurement helper, not artifact attestation.
The selected classifier executable is trusted, like the compiler and LLVM tools;
its hash identifies the executed bytes, not proof of how it was built. Build it
from the pinned standalone coverage-layout package as a separate source gate.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from coverage_binding import (
    MAX_EXPORT_BYTES, read_json, require_identity, validate_capture, write_json,
)
from coverage_lines import report_line_coverage
from coverage_provenance import (
    capture_sources, fingerprint_file, require_same_sources,
    validate_classification_sources,
)


MAX_CLASSIFIER_BYTES = 64 * 1024 * 1024
MAX_CLASSIFIER_LOG_BYTES = 8 * 1024 * 1024
MAX_CONTEXTS = 4096
MAX_ITEMS = 65_536
CAPTURE_PURPOSE = "fresh_whole_test_file_classifier_invocation"
POLICY = "whole_test_files_v2_exact_test_macro_spans_no_inline_or_generated_exclusions"


def relative_name(value: object) -> str:
    """Reject alternate root-relative spellings instead of normalizing aliases."""

    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("noncanonical classifier or compiler input name")
    return value


def target_entries(targets: list[dict], crate: Path) -> list[str]:
    """Select every distinct captured crate root, including integration contexts."""

    if not isinstance(targets, list) or not 3 <= len(targets) <= 512:
        raise ValueError("whole-file qualification needs captured production, unit and integration targets")
    entries, seen, production = set(), set(), []
    for item in targets:
        target = item["target"]
        if (not isinstance(target.get("name"), str) or not target["name"]
                or type(target.get("test")) is not bool
                or not isinstance(target.get("source"), str)):
            raise ValueError("invalid captured target identity")
        source = Path(target["source"])
        if not source.is_absolute() or source != source.resolve() or not source.is_relative_to(crate):
            raise ValueError("captured target source is not canonical and contained")
        key = (target["name"], str(source), target["test"])
        if key in seen:
            raise ValueError("duplicate captured target identity")
        seen.add(key)
        entries.add(relative_name(source.relative_to(crate).as_posix()))
        if not target["test"]:
            production.append(key)
    if len(production) != 1 or (*production[0][:2], True) not in seen:
        raise ValueError("whole-file qualification needs one production library and its unit target")
    return sorted(entries)


def capture_classifier(crate: Path, snapshot: dict, executable: Path, output: Path,
                       entries: list[str], *, timeout: float = 60) -> dict:
    """Run a selected trusted classifier once and retain original bytes create-only.

    Source and executable identities are checked before and after execution.
    Output must be outside the inventoried input roots. Trusted subprocesses can
    write to temporary streams until the timeout; byte limits bound subsequent
    reads, not hostile disk writes. Failures retain their record and native logs.
    """

    crate = crate.resolve(strict=True)
    executable = executable.absolute()
    if (not isinstance(entries, list) or not entries or any(not isinstance(entry, str) for entry in entries)
            or entries != sorted(set(entries))
            or any(relative_name(entry) not in snapshot["files"] for entry in entries)):
        raise ValueError("classifier requires explicit distinct captured entry files")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 60:
        raise ValueError("classifier timeout must be positive and at most 60 seconds")
    require_same_sources(snapshot, capture_sources(crate))
    identity = fingerprint_file(executable)
    if not os.access(executable, os.X_OK):
        raise ValueError("classifier is not executable")
    output = output.absolute()
    # Do not let evidence production mutate the very snapshot being measured.
    for source_root in snapshot["scope_roots"]:
        if output.resolve().is_relative_to(crate / source_root):
            raise ValueError("classifier evidence output overlaps source inventory")
    output.mkdir(parents=True, exist_ok=False)
    stdout, stderr = output / "classification.json", output / "stderr.log"
    command = [str(executable), str(crate), *entries]
    record = {"schema_version": 1, "purpose": CAPTURE_PURPOSE, "capture_valid": False,
              "process_exit_code": None, "command": command, "cwd": str(crate),
              "entries": entries, "executable_identity": identity,
              "source_inventory_sha256": snapshot["inventory_sha256"]}
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            child = subprocess.run(command, cwd=crate, stdout=out, stderr=err,
                                   timeout=timeout, check=False)
        record["process_exit_code"] = child.returncode
        record["stdout"] = {"path": str(stdout), "identity": fingerprint_file(stdout, maximum_bytes=MAX_CLASSIFIER_BYTES)}
        record["stderr"] = {"path": str(stderr), "identity": fingerprint_file(stderr, maximum_bytes=MAX_CLASSIFIER_LOG_BYTES)}
        require_same_sources(snapshot, capture_sources(crate))
        require_identity(executable, identity)
        if child.returncode != 0 or record["stderr"]["identity"]["bytes"]:
            raise ValueError("classifier failed or emitted stderr")
        classification = read_json(stdout, MAX_CLASSIFIER_BYTES)
        validate_classification_sources(snapshot, classification)
        if classification.get("entries") != entries:
            raise ValueError("classifier entries differ from explicit command")
        record["capture_valid"] = True
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        record["capture_error"] = str(error)
        raise
    finally:
        write_json(output / "invocation.json", record)
    return record


def validated_classifier(capture_path: Path, snapshot: dict, entries: list[str]) -> dict:
    """Authenticate a retained invocation, its source scope and unchanged output."""

    record = read_json(capture_path)
    crate = Path(snapshot["crate_root"])
    if (type(record.get("schema_version")) is not int or record["schema_version"] != 1
            or record.get("purpose") != CAPTURE_PURPOSE or record.get("capture_valid") is not True
            or type(record.get("process_exit_code")) is not int or record["process_exit_code"] != 0
            or record.get("cwd") != str(crate) or record.get("entries") != entries
            or record.get("source_inventory_sha256") != snapshot["inventory_sha256"]):
        raise ValueError("unqualified or differently scoped classifier invocation")
    command = record.get("command")
    if (not isinstance(command, list) or len(command) != len(entries) + 2
            or any(not isinstance(value, str) for value in command)
            or command[1:] != [str(crate), *entries] or not Path(command[0]).is_absolute()):
        raise ValueError("classifier command differs from explicit source scope")
    require_identity(Path(command[0]), record["executable_identity"])
    for key, name, bound in (("stdout", "classification.json", MAX_CLASSIFIER_BYTES),
                             ("stderr", "stderr.log", MAX_CLASSIFIER_LOG_BYTES)):
        path = capture_path.absolute().parent / name
        if record[key]["path"] != str(path):
            raise ValueError("classifier output is not the invocation's exact sibling")
        require_identity(path, record[key]["identity"], bound)
    if record["stderr"]["identity"]["bytes"] != 0:
        raise ValueError("classifier emitted diagnostics")
    result = read_json(Path(record["stdout"]["path"]), MAX_CLASSIFIER_BYTES)
    validate_classification_sources(snapshot, result)
    if result.get("entries") != entries:
        raise ValueError("classifier report omitted or added source roots")
    return result


def _integer(value: object, minimum: int, maximum: int) -> bool:
    """Do not accept booleans as source coordinates or line counters."""

    return type(value) is int and minimum <= value <= maximum


def _span(span: object, identity: dict) -> tuple[int, int]:
    """Validate bounded byte coordinates without pretending to classify spans."""

    if not isinstance(span, dict):
        raise ValueError("classifier span is not an object")
    points = []
    for key in ("start", "end"):
        point = span.get(key)
        if (not isinstance(point, dict)
                or not _integer(point.get("byte"), 0, identity["bytes"])
                or not _integer(point.get("line"), 1, identity["lines"] + 1)
                or any(not _integer(point.get(column), 1, identity["bytes"] + 1)
                       for column in ("column", "unicode_column"))):
            raise ValueError("classifier span has invalid coordinates")
        points.append(point["byte"])
    if points[0] > points[1]:
        raise ValueError("classifier span is reversed")
    return tuple(points)


def _span_coordinates(span: dict) -> tuple[int, ...]:
    """Compare every already-validated coordinate, not just overlapping bytes.

    Macro diagnostics and parsed macro regions must identify the same complete
    item. Byte equality alone must not hide mismatched line or Unicode columns.
    """

    return tuple(span[point][coordinate] for point in ("start", "end")
                 for coordinate in ("byte", "line", "column", "unicode_column"))


def _candidate(entry: dict, identity: dict, diagnostics: list[dict]) -> bool:
    """Require complete file-wide test ownership in every parsed context.

    This deliberately does not infer ownership from a filename, a test function
    attribute, selected covered lines, or an item span inside a production file.
    Malformed structures fail; well-formed ambiguity simply retains the file.
    """

    contexts, runs = entry.get("contexts"), entry.get("lines")
    if not isinstance(contexts, list) or not 1 <= len(contexts) <= MAX_CONTEXTS:
        raise ValueError("classifier contexts are missing or exceed their bound")
    if not isinstance(runs, list) or len(runs) > MAX_ITEMS:
        raise ValueError("classifier line runs are missing or exceed their bound")
    candidate, next_line = True, 1
    for run in runs:
        if (not isinstance(run, dict) or not _integer(run.get("start_line"), 1, identity["lines"])
                or run["start_line"] != next_line
                or not _integer(run.get("end_line"), run["start_line"], identity["lines"])
                or run.get("classification") not in {"retained", "test_only", "mixed", "blank"}):
            raise ValueError("classifier line runs are incomplete, overlapping or malformed")
        next_line = run["end_line"] + 1
        candidate &= run["classification"] in {"test_only", "blank"}
    if next_line != identity["lines"] + 1:
        raise ValueError("classifier line runs do not cover the physical source")
    common_test_macros = None
    for context in contexts:
        if (not isinstance(context, dict) or not isinstance(context.get("module"), str)
                or not context["module"] or type(context.get("duplicate_context")) is not bool):
            raise ValueError("invalid classifier context identity")
        regions = context.get("regions")
        if not isinstance(regions, list) or not 1 <= len(regions) <= MAX_ITEMS:
            raise ValueError("classifier ownership regions are missing or excessive")
        test_macros = set()
        for index, region in enumerate(regions):
            if (not isinstance(region, dict) or type(region.get("test_only")) is not bool
                    or not isinstance(region.get("kind"), str)
                    or not _integer(region.get("depth"), 0, (1 << 64) - 1)):
                raise ValueError("invalid classifier ownership region")
            span = _span(region.get("span"), identity)
            if index == 0:
                if region["kind"] != "file" or region["depth"] != 0 or span != (0, identity["bytes"]):
                    raise ValueError("classifier context lacks exact whole-file fallback")
            elif region["depth"] == 0 or region["kind"] == "file":
                raise ValueError("classifier has ambiguous whole-file fallback")
            candidate &= region["test_only"]
            if region["test_only"] and region["kind"] == "macro" and span[0] < span[1]:
                test_macros.add(_span_coordinates(region["span"]))
        common_test_macros = test_macros if common_test_macros is None else common_test_macros & test_macros
    # Opacity does not undo independently proven ownership of a whole test file.
    # It does prevent attributing other expansion filenames to this file, and it
    # never supplies the missing whole-file proof for inline tests. The grouped
    # diagnostic also represents verbatim/unresolved syntax: only exact parsed
    # macro spans shared by EVERY context qualify, never the diagnostic alone.
    for issue in diagnostics:
        if issue["code"] == "duplicate_context":
            continue
        if issue["code"] != "opaque_macro_or_verbatim" or issue.get("span") is None:
            return False
        _span(issue["span"], identity)
        if _span_coordinates(issue["span"]) not in common_test_macros:
            return False
    return candidate


def _thresholds(rows: list[dict], count_key: str, covered_key: str) -> dict:
    """Use exact integer 90/95 comparisons without promoting this diagnostic."""

    result = {}
    for label, threshold in (("crate", 90), ("shared", 95)):
        selected = [row for row in rows if label == "crate" or not row["path"].startswith("src/models/")]
        count, covered = sum(row[count_key] for row in selected), sum(row[covered_key] for row in selected)
        if count == 0:
            raise ValueError("empty whole-file diagnostic denominator")
        result[label] = {"lines": count, "covered": covered, "percent": 100 * covered / count,
                         "minimum_percent": threshold, "meets_threshold": 100 * covered >= threshold * count}
    return result


def whole_file_diagnostic(snapshot: dict, classification: dict, targets: list[dict], native: dict) -> dict:
    """Join already-authenticated inputs and retain a per-file exclusion ledger.

    This pure helper checks data contracts but cannot establish that its objects
    came from a compiler. Use ``qualify_capture`` for artifact-bearing evidence.
    No generated or inline line exclusions are supported. Legacy LLVM summaries
    remain distinct from before/after unique physical-line diagnostics.
    """

    crate = Path(snapshot["crate_root"])
    entries = target_entries(targets, crate)
    validate_classification_sources(snapshot, classification)
    if classification.get("entries") != entries:
        raise ValueError("classifier did not inspect every captured source root")
    production, unit, contexts = set(), set(), {}
    production_target = next(item["target"] for item in targets if not item["target"]["test"])
    for item in targets:
        target, inputs = item["target"], item.get("inputs")
        if (not isinstance(inputs, list) or not inputs or any(not isinstance(name, str) for name in inputs)
                or inputs != sorted(set(inputs))
                or any(relative_name(name) not in snapshot["files"] for name in inputs)):
            raise ValueError("compiler input inventory is incomplete or unbound")
        if str(Path(target["source"]).relative_to(crate)) not in inputs:
            raise ValueError("compiler input inventory omits its root")
        is_unit = target["test"] and all(target[field] == production_target[field] for field in ("name", "source"))
        for name in inputs:
            contexts.setdefault(name, []).append(target)
            if not target["test"]:
                production.add(name)
            if is_unit:
                unit.add(name)
    diagnostics = classification.get("diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) > MAX_ITEMS:
        raise ValueError("classifier diagnostics are missing or excessive")
    by_file = {}
    for issue in diagnostics:
        if (not isinstance(issue, dict) or not isinstance(issue.get("code"), str) or not issue["code"]
                or not isinstance(issue.get("file"), str)):
            raise ValueError("invalid classifier diagnostic")
        relative_name(issue["file"])
        by_file.setdefault(issue["file"], []).append(issue)
    candidates = {}
    for entry in classification["files"]:
        candidates[entry["path"]] = _candidate(entry, snapshot["files"][entry["path"]], by_file.get(entry["path"], []))
    measured = report_line_coverage(native)
    native_entries = {entry["filename"]: entry for block in native["data"] for entry in block["files"]}
    rows, seen = [], set()
    all_inputs = set(contexts)
    for filename, physical in measured.items():
        path = (crate / filename).resolve()
        if not path.is_relative_to(crate / "src"):
            continue
        name = relative_name(path.relative_to(crate).as_posix())
        if name in seen:
            raise ValueError("canonical aliases duplicate a native source file")
        seen.add(name)
        if not name.endswith(".rs") or name not in snapshot["files"] or name not in all_inputs:
            raise ValueError("native source is not an authenticated compiler Rust input")
        identity = snapshot["files"][name]
        if any(run.stop - 1 > identity["lines"] for run in physical.runs):
            raise ValueError("native mapped line exceeds physical source bounds")
        legacy = native_entries[filename]["summary"]["lines"]
        count, covered = legacy["count"], legacy["covered"]
        if not _integer(count, 0, (1 << 63) - 1) or not _integer(covered, 0, count):
            raise ValueError("invalid native legacy summary counters")
        if name in production:
            reason = "retained_production_compiler_input"
        elif name not in unit:
            reason = "retained_not_a_captured_unit_input"
        elif name not in candidates:
            reason = "retained_unclassified_source"
        elif not candidates[name]:
            reason = "retained_unknown_mixed_or_non_test_file_ownership"
        else:
            reason = "excluded_whole_test_file_absent_from_production_inputs"
        excluded = reason.startswith("excluded_")
        rows.append({"path": name, "identity": identity, "compiler_contexts": contexts[name],
                     "reason": reason, "excluded": excluded,
                     "mapped_lines": physical.mapped_lines, "covered_lines": physical.covered_lines,
                     "removed_mapped_lines": physical.mapped_lines if excluded else 0,
                     "removed_covered_lines": physical.covered_lines if excluded else 0,
                     "legacy_count": count, "legacy_covered": covered})
    rows.sort(key=lambda row: row["path"])
    return {"schema_version": 1, "policy": POLICY, "whole_test_file_policy_validated": True,
            "production_only_exclusions_qualified": False, "production_only_measurement": False,
            "compiler_dependency_closure_verified": False, "release_gate_promotion": False,
            "scope": "captured configuration; whole test files only; inline tests and generated or ambiguous ownership retained",
            "legacy_native_aggregate": _thresholds(rows, "legacy_count", "legacy_covered"),
            "physical_before_exclusions": _thresholds(rows, "mapped_lines", "covered_lines"),
            "physical_whole_file_diagnostic": _thresholds([row for row in rows if not row["excluded"]], "mapped_lines", "covered_lines"),
            "files": rows}


def _validated_run_results(run_results: Path, coverage_log: Path, setup: dict,
                           snapshot: dict, binding: dict, crate: Path) -> dict:
    """Bind the exact log to the successful row written by the existing runner.

    A clean sibling is not evidence for the actual coverage command. Require
    that row's native/validation status, indexed log name and retained hash,
    command/input context, setup and independently revalidated artifact binding.
    Every read record is fingerprinted for a second check after measurement.
    """

    from check_release import require_same_coverage_context

    root = Path(setup["compiler_root"]).parent
    if (run_results.absolute() != root / "results.json"
            or Path(setup["compiler_root"]) != root / "compiler"
            or Path(setup["export_root"]) != root / "llvm-export"):
        raise ValueError("run results are outside the bound native capture scope")
    identities = {}

    def retained(name: str, maximum: int = 8 * 1024 * 1024) -> dict:
        """Read an exact known sibling and retain its original byte identity."""

        path = root / name
        identity = fingerprint_file(path, maximum_bytes=maximum)
        value = read_json(path, maximum)
        require_identity(path, identity, maximum)
        identities[str(path)] = identity
        return value

    results = retained("results.json")
    gates = results.get("gates")
    if (type(results.get("schema_version")) is not int or results["schema_version"] != 1
            or not isinstance(gates, list) or not 1 <= len(gates) <= 512
            or any(not isinstance(row, dict) for row in gates)):
        raise ValueError("invalid retained release-run result schema")
    selected = [(index, row) for index, row in enumerate(gates) if row.get("gate") == "coverage"]
    if len(selected) != 1:
        raise ValueError("run results need exactly one coverage command row")
    index, row = selected[0]
    if (any(type(row.get(field)) is not int or row[field] != 0
            for field in ("exit_code", "process_exit_code")) or "validation_error" in row):
        raise ValueError("retained coverage command did not succeed natively and validate")
    log_name = f"{index:02d}-coverage.log"
    if row.get("log") != log_name or coverage_log.absolute() != root / log_name:
        raise ValueError("coverage log is not the actual retained command log")
    require_identity(coverage_log, row["log_identity"], 512 * 1024 * 1024)
    identities[str(coverage_log)] = row["log_identity"]
    compiler = row.get("compiler_evidence")
    inputs = row.get("input_evidence")
    if (not isinstance(compiler, dict) or compiler.get("setup") != "compiler-capture-setup.json"
            or compiler.get("binding") != "compiler-binding.json"
            or any(compiler.get(field) is not True for field in (
                "artifact_binding_verified", "compiler_reported_inputs_verified", "no_export_filter_verified"))
            or not isinstance(inputs, dict) or inputs.get("before") != "coverage-inputs-before.json"
            or inputs.get("after") != "coverage-inputs-after.json" or inputs.get("sources_unchanged") is not True):
        raise ValueError("coverage row lacks the required native setup and source evidence")
    if retained("compiler-capture-setup.json") != setup:
        raise ValueError("coverage row setup differs from the revalidated compiler setup")
    original_binding = retained("compiler-binding.json")
    if any(original_binding.get(key) != value for key, value in binding.items()):
        raise ValueError("coverage row binding differs from independently revalidated artifacts")
    before, after = retained("coverage-inputs-before.json"), retained("coverage-inputs-after.json")
    require_same_coverage_context(before, after)
    command = row.get("command")
    if (before["sources"] != snapshot or before.get("working_directory") != str(crate)
            or command != before.get("command") or not isinstance(command, list)
            or any(not isinstance(value, str) for value in command)
            or len(command) < 3 or command[0] != "cargo" or not command[1].startswith("+")
            or command[2] != "llvm-cov" or command.count("--disable-default-ignore-filename-regex") != 1):
        raise ValueError("coverage command differs from its retained input context")
    environment = before.get("compiler_environment", {})
    if (environment.get("TFM_COVERAGE_CAPTURE_ROOT") != setup["compiler_root"]
            or environment.get("TFM_COVERAGE_ARTIFACT_ROOT") != setup["artifact_root"]
            or environment.get("RUSTC_WORKSPACE_WRAPPER") != str(root / "rustc-workspace-wrapper")):
        raise ValueError("coverage command input context refers to another capture setup")
    report = root / "coverage.json"
    require_identity(report, row["report_identity"], MAX_EXPORT_BYTES)
    identities[str(report)] = row["report_identity"]
    return identities


def qualify_capture(setup: dict, snapshot: dict, capture_path: Path, crate: Path,
                    *, coverage_log: Path, run_results: Path) -> dict:
    """Revalidate native evidence and qualify only the explicitly scoped file join.

    No caller-provided boolean can substitute for actual artifact revalidation.
    The source snapshot, classifier record/output, and native report are checked
    again around measurement. This detects ordinary edits, not malicious tools
    or edit-and-restore races. Full production-only qualification stays false.
    """

    # Import lazily so the existing runner can later consume this module without
    # a module-initialization cycle. Its exact diagnostic policy stays singular.
    from check_release import read_coverage_log, require_clean_coverage_log

    crate = crate.resolve(strict=True)
    require_same_sources(snapshot, capture_sources(crate))
    binding = validate_capture(setup, snapshot, crate)
    run_identities = _validated_run_results(run_results, coverage_log, setup, snapshot, binding, crate)
    log_identity = fingerprint_file(coverage_log, maximum_bytes=512 * 1024 * 1024)
    require_clean_coverage_log(read_coverage_log(coverage_log), allow_unfiltered_option_notice=(
        binding["no_export_filter_verified"] is True))
    entries = target_entries(binding["targets"], crate)
    capture_identity = fingerprint_file(capture_path)
    classification = validated_classifier(capture_path, snapshot, entries)
    native_path = Path(binding["native_report"]["path"])
    native = read_json(native_path, MAX_EXPORT_BYTES)
    result = whole_file_diagnostic(snapshot, classification, binding["targets"], native)
    require_identity(native_path, binding["native_report"]["identity"], MAX_EXPORT_BYTES)
    validated_classifier(capture_path, snapshot, entries)
    require_identity(capture_path, capture_identity)
    require_identity(coverage_log, log_identity, 512 * 1024 * 1024)
    for path, identity in run_identities.items():
        require_identity(Path(path), identity, MAX_EXPORT_BYTES)
    require_same_sources(snapshot, capture_sources(crate))
    result["whole_test_file_exclusions_qualified"] = True
    result["classifier_capture"] = {"path": str(capture_path), "identity": capture_identity}
    result["native_report"] = binding["native_report"]
    result["coverage_log"] = {"path": str(coverage_log), "identity": log_identity}
    result["retained_run_evidence"] = run_identities
    result["source_inventory_sha256"] = snapshot["inventory_sha256"]
    return result
