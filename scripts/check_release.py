"""Run explicit, local source-release gates without downloading model artifacts.

Commands execute without a shell and stop at the first failure. Logs and the
command manifest are written to a new directory below Cargo's normal target
directory; no source files, Git refs, checkpoints, or support flags are changed.
Hardware acceptance and fuzz campaigns remain separate, serialized workflows.

The ``verus`` gate runs every proof harness with the pinned Verus release named
by the ``VERUS`` environment variable and requires each harness's verified
count to equal the one recorded in ``verus/README.md``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tomllib

from coverage_provenance import capture_sources, fingerprint_file, require_same_sources
from coverage_binding import prepare_capture, validate_capture, read_json, require_identity
import package_contents


CRATE = Path(__file__).resolve().parents[1]
FEATURES = "std,flex,tokenizers,chat-templates,autodiff"
GATES = ("source", "msrv", "coverage", "dependencies", "accelerator-compile", "package", "verus", "guides")
MINIMUM_BUILD_FREE_BYTES = 20 * 1024**3
# crates.io rejects uploads above 10 MiB. The package gate fails well before
# that, so ordinary growth surfaces as a decision about what to publish rather
# than as a rejected upload (see package_contents.py for the evidence split).
PACKAGE_MAX_BYTES = 8 * 1024**2
# Public experiments with fixed/random fixtures and numerical assertions. Paper
# checkpoints have separate explicit preparation/replay protocols; never fetch
# artifacts as a side effect of checking source code.
SOURCE_EXPERIMENTS = (
    ("activation_workbench", ()),
    ("attention_head_attribution", ()),
    ("custom_composition", ()),
    ("t5_classifier", ()),
    ("mla_encoder_decoder", ()),
    ("qwen_prefix", ("--random",)),
)
# These are maintenance notices, not vulnerability waivers. New IDs fail.
MAINTENANCE_NOTICES = ("RUSTSEC-2025-0141", "RUSTSEC-2024-0436")
UNFILTERED_OPTION_NOTICE = "warning: --disable-default-ignore-filename-regex option is unstable"
RUNNER_INPUTS = tuple(Path(__file__).resolve().with_name(name) for name in (
    "check_release.py", "coverage_provenance.py", "coverage_binding.py", "coverage_compiler.py", "coverage_dependencies.py",
))
# The Verus release that verus/README.md and the CI workflow pin.
VERUS_VERSION = "0.2026.10.04.426d8b0"
# Every harness is a library crate root; --no-cheating rejects assume/admit,
# external_body, and assume_specification in the prover itself.
VERUS_FLAGS = ("--crate-type", "lib", "--no-cheating", "--triggers-mode", "silent")
VERUS_RESULT = re.compile(r"^verification results:: (\d+) verified, (\d+) errors?\s*$", re.MULTILINE)


def coverage_tool_versions(command: list[str], crate: Path, environment: dict) -> list[dict]:
    """Record the selected compiler/Cargo/exporter versions, not binary attestation."""

    if len(command) < 3 or command[0] != "cargo" or not command[1].startswith("+") or command[2] != "llvm-cov":
        raise ValueError("coverage command has no explicit Cargo toolchain")
    toolchain = command[1]
    commands = (["rustc", toolchain, "--version", "--verbose"],
                ["cargo", toolchain, "--version"], ["cargo", toolchain, "llvm-cov", "--version"])
    versions = []
    for invocation in commands:
        output = subprocess.check_output(invocation, cwd=crate, env=environment,
                                         stderr=subprocess.STDOUT, text=True, timeout=30)
        require_clean_coverage_log(output)
        if not output.strip():
            raise ValueError(f"empty coverage tool identity: {invocation[0]}")
        versions.append({"command": invocation, "output": output})
    return versions


def capture_coverage_context(crate: Path, command: list[str], environment: dict) -> dict:
    """Bind declared source and runner bytes, without claiming compiler closure.

    Only explicit compiler-control environment variables are retained. This is
    an ordinary local reproducibility record, not an attestation that wrappers,
    build scripts, procedural macros or inherited Cargo settings are hermetic.
    """

    return {
        "schema_version": 1,
        "sources": capture_sources(crate),
        "runner_inputs": {path.name: fingerprint_file(path) for path in RUNNER_INPUTS},
        "tool_versions": coverage_tool_versions(command, crate, environment),
        "command": command,
        "working_directory": str(crate.resolve()),
        "compiler_environment": {name: environment[name] for name in (
            "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_INCREMENTAL",
            "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR", "RUSTC", "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "LLVM_COV", "LLVM_PROFDATA", "TFM_COVERAGE_CAPTURE_ROOT", "TFM_COVERAGE_ARTIFACT_ROOT",
        ) if name in environment},
        "compiler_dependency_closure_verified": False,
        "production_only_exclusions_qualified": False,
    }


def require_same_coverage_context(before: dict, after: dict) -> None:
    """Reject input edits before consulting any native coverage counters."""

    require_same_sources(before["sources"], after["sources"])
    for key in ("runner_inputs", "tool_versions", "command", "working_directory", "compiler_environment"):
        if before[key] != after[key]:
            raise ValueError(f"coverage context changed: {key}")


def write_evidence(path: Path, value: dict) -> None:
    """Retain one human-readable JSON observation inside a fresh run directory."""

    path.write_text(json.dumps(value, indent=2) + "\n")


def read_coverage_log(path: Path) -> str:
    """Read bounded literal log bytes, preserving CRLF and exact notice spelling."""

    maximum = 512 * 1024 * 1024
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("coverage log exceeds its byte bound")
    return payload.decode("utf-8")


def require_clean_coverage_log(log: str, *, allow_unfiltered_option_notice: bool = False) -> int:
    """Reject compiler/exporter warnings even when the native process exits zero.

    LLVM may export partial coverage after reporting mismatched function data.
    Such counters must not enter a gate. The explicit unfiltered binding mode
    may retain exactly one byte-exact pinned cargo-llvm-cov option notice, only
    after proving the export contains no filter. It is not a numerical or
    compiler-warning waiver. Legacy mode still rejects every diagnostic.
    ANSI styling cannot hide diagnostics or qualify a variant notice.
    """

    diagnostic = re.compile(r"^\s*(?:.*?:\s+)?(?:warning|error)(?:\[[^]\n]+\])?:(?:\s|$)", re.IGNORECASE)
    notices = 0
    for line in log.split("\n"):
        if allow_unfiltered_option_notice and line == UNFILTERED_OPTION_NOTICE and notices == 0:
            notices += 1
            continue
        plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
        if diagnostic.match(plain):
            raise ValueError(f"unqualified coverage diagnostic: {plain.strip()}")
    if allow_unfiltered_option_notice and notices != 1:
        raise ValueError("unfiltered coverage did not emit its expected pinned option notice")
    return notices


def coverage_source_relative(filename: str, crate: Path) -> str | None:
    """Resolve LLVM filenames against the command cwd, retaining only src inputs."""

    if not isinstance(filename, str) or not filename or "\0" in filename:
        raise ValueError("invalid coverage filename")
    root = crate.resolve()
    path = Path(filename)
    path = (path if path.is_absolute() else root / path).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root / "src") else None


def require_bound_coverage_sources(report: dict, snapshot: dict, crate: Path) -> None:
    """Reject demonstrably stale src entries without claiming complete closure."""

    for data in report["data"]:
        for entry in data["files"]:
            relative = coverage_source_relative(entry["filename"], crate)
            if relative is not None and (not relative.endswith(".rs") or relative not in snapshot["files"]):
                raise ValueError(f"coverage source is not a captured Rust input: {relative}")


def coverage_summary(report: dict, crate: Path) -> dict:
    """Aggregate reported LLVM line counters without additional src exclusions.

    Inline unit-test bodies and macro expansions are included by LLVM. This
    deliberately makes no production-only or branch-coverage claim. Integration
    harness files are not part of either denominator; experimental model source
    remains in the crate-wide denominator. Both groups must meet their gate.
    """

    files = []
    seen = set()
    for data in report["data"]:
        for entry in data["files"]:
            relative = coverage_source_relative(entry["filename"], crate)
            if relative is None:
                continue
            if relative in seen:
                raise ValueError(f"duplicate coverage file: {relative}")
            seen.add(relative)
            lines = entry["summary"]["lines"]
            count, covered = lines["count"], lines["covered"]
            if type(count) is not int or type(covered) is not int or not 0 <= covered <= count:
                raise ValueError(f"invalid coverage counters: {relative}")
            files.append((relative, count, covered))
    result = {}
    for label, threshold in (("crate", 90), ("shared", 95)):
        selected = [row for row in files if label == "crate" or not row[0].startswith("src/models/")]
        count = sum(row[1] for row in selected)
        covered = sum(row[2] for row in selected)
        if count == 0:
            raise ValueError(f"empty coverage denominator: {label}")
        result[label] = {
            "lines": count, "covered": covered, "percent": 100 * covered / count,
            "minimum_percent": threshold, "passed": 100 * covered >= threshold * count,
        }
    result["scope"] = "reported src/**/*.rs after exporter directory filters, including inline unit tests and generated expansions"
    result["branch_coverage_measured"] = False
    result["passed"] = all(result[label]["passed"] for label in ("crate", "shared"))
    return result


def cargo_target_directory() -> Path:
    """Return Cargo's target directory for commands that run in this crate.

    Cargo resolves a relative ``CARGO_TARGET_DIR`` against its working
    directory, which is the crate root for every gate command.
    """

    configured = os.environ.get("CARGO_TARGET_DIR")
    return CRATE / configured if configured else CRATE / "target"


def package_stem() -> str:
    """Return ``NAME-VERSION``, the stem Cargo gives this crate's archive and its extraction."""

    package = tomllib.loads((CRATE / "Cargo.toml").read_text())["package"]
    return f"{package['name']}-{package['version']}"


def package_directory() -> Path:
    """Return the directory where ``cargo package`` extracts and verifies the archive."""

    return cargo_target_directory() / "package" / package_stem()


def package_archive() -> Path:
    """Return the compressed ``.crate`` archive that ``cargo package`` writes."""

    return cargo_target_directory() / "package" / f"{package_stem()}.crate"


def measure_package_archive(archive: Path, maximum_bytes: int = PACKAGE_MAX_BYTES) -> dict:
    """Identify the built archive and fail when it exceeds the upload budget.

    Returns the archive's path, size, SHA-256 and the budget, which the gate
    manifest retains beside the archive. Raises ``ValueError`` for a missing
    or non-regular archive and for one larger than ``maximum_bytes``.
    """

    if archive.is_symlink() or not archive.is_file():
        raise ValueError(f"package archive is not a regular file: {archive}")
    size = archive.stat().st_size
    if size > maximum_bytes:
        raise ValueError(f"package archive {archive} is {size} bytes, above the {maximum_bytes}-byte "
                         "budget below crates.io's 10 MiB upload limit; keep bulky evidence repository-only")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(archive), "bytes": size, "sha256": digest, "maximum_bytes": maximum_bytes}


def guide_doctests(markdown: str) -> tuple[str, int]:
    """Rewrite one guide so rustdoc compiles every Rust block as a doctest.

    A complete block (items, or statements that need nothing undeclared)
    compiles as written. A fragment that assumes a model, an input, or other
    surrounding code is preceded by ``<!-- doctest NAME -->`` and compiled
    inside the context that an HTML comment of the same guide defines::

        <!-- doctest-context NAME
        use ...;
        fn fragment<B: Backend>(...) -> transformers::Result<()> {
            __FRAGMENT__
            Ok(())
        }
        -->

    HTML comments are invisible in rendered Markdown, so the context lines
    stay out of the reader's way; the rewrite turns them into rustdoc hidden
    lines (``# …``) around the fragment. Blocks fenced as ``rust,ignore`` are
    deliberately not compiled. An unknown or unused context name fails, so a
    stale marker cannot silently stop a fragment from compiling. Returns the
    rewritten Markdown and the number of Rust blocks that will compile.
    """

    contexts = {}
    for match in re.finditer(r"^<!-- doctest-context ([a-z0-9-]+)\n(.*?)^-->\n", markdown, re.MULTILINE | re.DOTALL):
        name, body = match.group(1), match.group(2).rstrip("\n").split("\n")
        if name in contexts or sum(line.strip() == "__FRAGMENT__" for line in body) != 1:
            raise ValueError(f"doctest context {name!r} must be unique and contain one __FRAGMENT__ line")
        contexts[name] = body
    text = re.sub(r"^<!-- doctest-context [a-z0-9-]+\n.*?^-->\n", "", markdown, flags=re.MULTILINE | re.DOTALL)
    lines, output, used, compiled = text.split("\n"), [], set(), 0
    index = 0
    while index < len(lines):
        line = lines[index]
        fence = re.match(r"^(\s*)```(.*)$", line)
        if not fence:
            output.append(line)
            index += 1
            continue
        info = fence.group(2).strip()
        end = next((position for position in range(index + 1, len(lines))
                    if lines[position].strip() == "```"), None)
        if end is None:
            raise ValueError(f"unclosed code fence at line {index + 1}")
        body = lines[index + 1:end]
        language = info.split(",")[0].strip()
        previous = next((output[position] for position in range(len(output) - 1, -1, -1)
                         if output[position].strip()), "")
        marker = re.fullmatch(r"<!-- doctest ([a-z0-9-]+) -->", previous.strip())
        if marker and language != "rust":
            raise ValueError(f"doctest marker before a non-Rust block at line {index + 1}")
        if language == "rust" and "ignore" not in info.split(","):
            compiled += 1
            if marker:
                name = marker.group(1)
                if name not in contexts:
                    raise ValueError(f"unknown doctest context {name!r} at line {index + 1}")
                used.add(name)
                wrapped = []
                for context_line in contexts[name]:
                    if context_line.strip() == "__FRAGMENT__":
                        wrapped.extend(body)
                    else:
                        wrapped.append(f"# {context_line}" if context_line else "#")
                body = wrapped
        # rustdoc compiles an unlabeled fence as Rust; label it as text instead.
        output.extend([fence.group(1) + "```" + (info or "text"), *body, lines[end]])
        index = end + 1
    unused = sorted(set(contexts) - used)
    if unused:
        raise ValueError(f"unused doctest contexts: {unused}")
    return "\n".join(output), compiled


GUIDES = ("README.md", "EXPERIMENTS.md", "TRAINING.md", "V0_2_INTERFACE.md")


def write_guide_doctests(crate: Path, directory: Path, offline: bool) -> dict:
    """Write a scratch crate whose doctests are the Rust blocks of the guides.

    The crate depends on this one by path with the source gate's features, as
    a downstream user would, and starts from a copy of this crate's lockfile.
    ``cargo metadata`` then adds only the scratch package, so the checked
    command can run with ``--locked`` against the same dependency versions.
    """

    directory.mkdir(parents=True)
    (directory / "guides").mkdir()
    features = ", ".join(f'"{feature}"' for feature in f"{FEATURES},train".split(","))
    manifest = tomllib.loads((crate / "Cargo.toml").read_text())
    burn = manifest["dependencies"]["burn"]["version"]
    (directory / "Cargo.toml").write_text(
        "[package]\nname = \"transformers-guide-doctests\"\nversion = \"0.0.0\"\nedition = \"2024\"\n"
        "publish = false\n\n[lib]\npath = \"lib.rs\"\n\n[dependencies]\n"
        f"burn = {{ version = \"{burn}\", default-features = false }}\n"
        f"transformers = {{ path = {json.dumps(str(crate.resolve()))}, default-features = false, "
        f"features = [{features}] }}\n\n[workspace]\n"
    )
    items, counts = [], {}
    for guide in GUIDES:
        rewritten, counts[guide] = guide_doctests((crate / guide).read_text())
        (directory / "guides" / guide).write_text(rewritten)
        name = "".join(part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", guide.removesuffix(".md")) if part)
        items.append(f"/// Doctests generated from `{guide}`.\n#[doc = include_str!(\"guides/{guide}\")]\npub struct {name};\n")
    (directory / "lib.rs").write_text(
        "//! The Rust blocks of the repository guides, compiled as doctests by\n"
        "//! `scripts/check_release.py --gate guides`. Generated; do not edit.\n\n" + "\n".join(items)
    )
    shutil.copyfile(crate / "Cargo.lock", directory / "Cargo.lock")
    subprocess.run(["cargo", "metadata", "--format-version", "1", "--manifest-path", str(directory / "Cargo.toml")]
                   + (["--offline"] if offline else []), check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, text=True)
    return {"guides": counts, "compiled_blocks": sum(counts.values())}


def verus_harnesses(crate: Path) -> list[Path]:
    """List every Verus crate root, each ``verus/*.rs`` file, in file-name order."""

    return sorted(path for path in (crate / "verus").glob("*.rs") if path.is_file())


def verus_included_sources(harness: Path) -> list[str]:
    """Return the ``src/`` files a harness verifies through ``#[path = "…"]`` attributes.

    Nested inline modules change how Rust resolves relative paths, so only the
    suffix starting at ``src/`` names the file, as in ``tests/verus_harness.rs``.
    """

    sources = []
    for target in re.findall(r'#\[path = "([^"]+)"\]', harness.read_text()):
        offset = target.find("src/")
        if offset < 0:
            raise ValueError(f"{harness.name}: #[path] must name a file under src/: {target}")
        sources.append(target[offset:])
    return sources


def recorded_verus_counts(readme: str) -> tuple[dict[str, int], int]:
    """Read the harness table and bold total of the ``Harnesses`` section of verus/README.md.

    Each row reads ``| `name.rs` | sources | verified | proofs |``; the total
    reads ``**N verified, 0 errors**``. Both are what the last recorded run
    reported, and the gate compares a new run against them.
    """

    start = readme.find("## Harnesses\n")
    if start < 0:
        raise ValueError("verus/README.md has no Harnesses section")
    section = readme[start:].split("\n## ", 1)[0]
    counts = {}
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = line.split("|")
        if len(cells) < 5:
            raise ValueError(f"malformed verus/README.md harness row: {line}")
        name, verified = cells[1].strip().strip("`"), cells[3].strip().replace(",", "")
        if not verified.isdigit() or name in counts:
            raise ValueError(f"malformed verus/README.md harness row: {line}")
        counts[name] = int(verified)
    total = re.search(r"\*\*([\d,]+) verified, 0 errors\*\*", section)
    if not counts or total is None:
        raise ValueError("verus/README.md records no harness counts or total")
    return counts, int(total.group(1).replace(",", ""))


def parse_verus_result(log: str) -> int:
    """Return one harness run's verified count; anything but one error-free result fails."""

    results = VERUS_RESULT.findall(log)
    if len(results) != 1:
        raise ValueError(f"expected one Verus result line, found {len(results)}")
    verified, errors = (int(value) for value in results[0])
    if errors:
        raise ValueError(f"Verus reported {errors} errors")
    return verified


def require_pinned_verus(version_output: str) -> None:
    """Reject a ``$VERUS`` that is not the release the proofs are recorded against."""

    if not re.search(rf"^\s*Version:\s*{re.escape(VERUS_VERSION)}\s*$", version_output, re.MULTILINE):
        raise ValueError(f"$VERUS is not the pinned Verus release {VERUS_VERSION}")


def capture_verus_inputs(crate: Path) -> dict:
    """Fingerprint every harness and each source it includes (SHA-256, bytes, lines)."""

    return {harness.name: {
        "harness": fingerprint_file(harness),
        "sources": {source: fingerprint_file(crate / source) for source in verus_included_sources(harness)},
    } for harness in verus_harnesses(crate)}


def start_verus_evidence(crate: Path) -> dict:
    """Read the recorded counts and fingerprint the proof inputs before any harness runs."""

    recorded, total = recorded_verus_counts((crate / "verus/README.md").read_text())
    names = [harness.name for harness in verus_harnesses(crate)]
    if sorted(recorded) != names:
        raise ValueError(f"verus/README.md lists {sorted(recorded)}, but verus/ holds {names}")
    return {"recorded": recorded, "recorded_total": total, "measured": {},
            "inputs_before": capture_verus_inputs(crate)}


def record_verus_step(state: dict, command: list[str], log: str) -> dict:
    """Validate one successful Verus command and return the fields its manifest row records."""

    if command[1:] == ["--version"]:
        require_pinned_verus(log)
        return {"verus_version": VERUS_VERSION}
    name = Path(command[-1]).name
    verified = parse_verus_result(log)
    recorded = state["recorded"].get(name)
    if verified != recorded:
        raise ValueError(f"{name}: Verus verified {verified} items, but verus/README.md records {recorded}; "
                         "update its harness table and total")
    state["measured"][name] = verified
    return {"harness": name, "verified": verified}


def finish_verus_evidence(state: dict, crate: Path) -> dict:
    """Require unchanged inputs and the recorded total, then return the gate's evidence."""

    after = capture_verus_inputs(crate)
    if after != state["inputs_before"]:
        raise ValueError("a Verus harness or included source changed while the gate ran")
    if sorted(state["measured"]) != sorted(state["recorded"]):
        raise ValueError("not every recorded harness was verified")
    total = sum(state["measured"].values())
    if total != state["recorded_total"]:
        raise ValueError(f"Verus verified {total} items, but verus/README.md records {state['recorded_total']}")
    return {"schema_version": 1, "verus_version": VERUS_VERSION, "flags": list(VERUS_FLAGS),
            "harnesses": state["measured"], "verified": total, "errors": 0, "inputs": after}


def command_plan(gate: str, output: Path, toolchain: str, msrv: str, offline: bool,
                 verus: str | None = None) -> list[list[str]]:
    """Build auditable argument arrays; platform checks never execute GPU code.

    ``verus`` is the Verus executable that the ``verus`` gate requires.
    """

    cargo = ["cargo", f"+{toolchain}"]
    manifest = ["--manifest-path", str(CRATE / "Cargo.toml")]
    locked = ["--locked"] + (["--offline"] if offline else [])
    feature_args = ["--no-default-features", "--features", FEATURES]
    layout_manifest = ["--manifest-path", str(CRATE / "scripts/coverage-layout/Cargo.toml")]
    # Both standalone packages share the runtime crate's normal target location.
    # This avoids creating a nested build directory inside packaged tool sources.
    layout_target = ["--target-dir", str(CRATE / "target")]
    if gate == "source":
        # Compile and exercise the optional native Burn Learner integration.
        # Keep the separately bound coverage feature profile explicit above.
        feature_args = ["--no-default-features", "--features", f"{FEATURES},train"]
        return [
            cargo + ["fmt"] + manifest + ["--check"],
            cargo + ["clippy"] + manifest + locked + ["--all-targets"] + feature_args + ["--", "-D", "warnings"],
            cargo + ["test"] + manifest + locked + feature_args + ["--all-targets"],
            cargo + ["test"] + manifest + locked + feature_args + ["--doc"],
            cargo + ["check"] + manifest + locked + ["--no-default-features"],
            cargo + ["check"] + manifest + locked + ["--no-default-features", "--features", "std"],
            cargo + ["doc"] + manifest + locked + feature_args + ["--no-deps"],
            [sys.executable, "-m", "unittest", "discover", "-s", str(CRATE / "scripts/tests"), "-v"],
            cargo + ["fmt"] + layout_manifest + ["--check"],
            cargo + ["clippy"] + layout_manifest + locked + layout_target + ["--all-targets", "--", "-D", "warnings"],
            cargo + ["test"] + layout_manifest + locked + layout_target,
            cargo + ["build"] + layout_manifest + locked + layout_target,
            [sys.executable, str(CRATE / "scripts/check_documentation.py"),
             "--standards",
             "--classifier", str(CRATE / "target/debug/transformers-coverage-layout"),
             "--output", str(output / "documentation.json")],
            *[
                cargo + ["run", "--release"] + manifest + locked + feature_args
                + ["--example", name] + (["--", *arguments] if arguments else [])
                for name, arguments in SOURCE_EXPERIMENTS
            ],
            cargo + ["run", "--release"] + manifest + locked + feature_args
            + ["--example", "burn_training", "--", str(output / "burn_training")],
        ]
    if gate == "msrv":
        return [
            ["cargo", f"+{msrv}", "test"] + manifest + locked + feature_args,
            ["cargo", f"+{msrv}", "test"] + layout_manifest + locked + layout_target,
        ]
    if gate == "coverage":
        return [cargo + ["llvm-cov"] + manifest + locked + feature_args + [
            "--lib", "--tests", "--json", "--output-path", str(output / "coverage.json"),
        ]]
    if gate == "dependencies":
        # cargo-audit checks every lockfile entry for vulnerabilities, including
        # optional Burn backends that no feature activates; cargo-deny judges
        # advisories and yanked versions over the graph this crate builds.
        audit = ["cargo", "audit", "--file", str(CRATE / "Cargo.lock"), "--deny", "warnings", "--no-yanked"]
        if offline:
            audit += ["--no-fetch"]
        for advisory in MAINTENANCE_NOTICES:
            audit += ["--ignore", advisory]
        # `locked` carries `--offline`, which also stops cargo-deny fetching its database.
        deny = ["cargo", "deny"] + manifest + locked + ["--config", str(CRATE / "deny.toml"), "check"]
        return [audit, deny + ["advisories", "licenses", "sources"]]
    if gate == "accelerator-compile":
        return [
            cargo + ["check"] + manifest + locked
            + ["--no-default-features", "--features", f"std,{backend},tokenizers,chat-templates"]
            for backend in ("metal", "cuda", "rocm")
        ]
    if gate == "package":
        # The parent repository may contain unrelated changes. Package listing
        # is diagnostic; the verified final archive still requires clean Git.
        # Cargo's verification only builds the extracted archive, so its test
        # suite runs there too: tests must not read files the archive omits.
        # Like the verification build, the tests share Cargo's normal target
        # directory; otherwise every dependency would be rebuilt inside the
        # archive directory, which the next `cargo package` deletes. The
        # packaged Python tests run there as well: record tests that need
        # repository-only evidence must skip, not fail (package_contents.py).
        return [
            cargo + ["package"] + manifest + locked + ["--list"],
            cargo + ["package"] + manifest + locked,
            cargo + ["test", "--manifest-path", str(package_directory() / "Cargo.toml")] + locked
            + ["--target-dir", str(cargo_target_directory())],
            [sys.executable, "-m", "unittest", "discover", "-s", str(package_directory() / "scripts/tests"), "-v"],
        ]
    if gate == "guides":
        # write_guide_doctests prepares this scratch crate before the command
        # runs; sharing the target directory reuses the source gate's build.
        return [cargo + ["test", "--doc", "--manifest-path", str(output / "guide-doctests/Cargo.toml")]
                + locked + ["--target-dir", str(cargo_target_directory())]]
    if gate == "verus":
        if not verus:
            raise ValueError("the verus gate needs the VERUS environment variable to name the pinned "
                             f"Verus {VERUS_VERSION} executable (see verus/README.md)")
        return [[verus, "--version"]] + [
            [verus, *VERUS_FLAGS, str(harness)] for harness in verus_harnesses(CRATE)
        ]
    raise ValueError(f"unknown gate: {gate}")


def command_compiles(command: list[str]) -> bool:
    """Identify compile-capable invocations in the explicit release command plan.

    Rustdoc, examples, coverage, and package verification can compile even when
    an ordinary build is not requested. Formatting and package listing cannot.
    """

    if not command or command[0] != "cargo" or len(command) < 2:
        return False
    index = 2 if command[1].startswith("+") else 1
    if index >= len(command):
        return False
    operation = command[index]
    return operation in {"build", "check", "clippy", "doc", "test", "run", "llvm-cov", "package"} and not (
        operation == "package" and "--list" in command
    )


def observe_build_storage(crate: Path, environment: dict) -> dict:
    """Measure local free bytes without cleaning artifacts or selecting a fallback.

    This checks the checkout's local filesystem immediately before compilation.
    Explicit Cargo target overrides are retained as provenance, not silently
    changed or treated as evidence that the local-space requirement was met.
    An unreadable measurement fails closed with its diagnostic in the record.
    """

    observation = {
        "checked_path": str(crate.resolve()),
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "minimum_free_bytes": MINIMUM_BUILD_FREE_BYTES,
        "free_bytes": None,
        "passed": False,
        "target_environment": {name: environment[name] for name in (
            "CARGO_TARGET_DIR", "CARGO_LLVM_COV_TARGET_DIR",
        ) if name in environment},
    }
    try:
        observation["free_bytes"] = shutil.disk_usage(crate).free
    except OSError as error:
        observation["measurement_error"] = str(error)
    else:
        observation["passed"] = observation["free_bytes"] >= MINIMUM_BUILD_FREE_BYTES
    return observation


def require_msrv(toolchain: str) -> None:
    """Reject an alias that no longer resolves to the declared Rust 1.92 MSRV."""

    version = subprocess.check_output(["rustc", f"+{toolchain}", "--version"], text=True).split()
    if len(version) < 2 or version[1] != "1.92.0":
        raise ValueError(f"MSRV alias {toolchain!r} resolves to {' '.join(version)}, not rustc 1.92.0")


# Committed tiny random-weight goldens that the packaged tests read. They are
# reference fixtures, not checkpoints: one directory per case, each file a
# regular file no larger than the cap, so real weights still fail the listing.
TINY_GOLDEN_FIXTURES = Path("tests/fixtures/tiny_goldens")
TINY_GOLDEN_MAX_BYTES = 1024 * 1024


def is_tiny_golden_fixture(path: Path, root: Path) -> bool:
    """Whether a listed weight file is a committed tiny golden the archive's tests read.

    Only ``tests/fixtures/tiny_goldens/<case>/<name>.safetensors`` qualifies,
    and only as a regular file (not a symlink) of at most
    ``TINY_GOLDEN_MAX_BYTES`` below ``root``.
    """

    candidate = root / path
    return (path.suffix == ".safetensors" and path.parent.parent == TINY_GOLDEN_FIXTURES
            and not candidate.is_symlink() and candidate.is_file()
            and candidate.stat().st_size <= TINY_GOLDEN_MAX_BYTES)


def validate_package_listing(listing: str, root: Path = CRATE) -> int:
    """Fail before packaging if Cargo follows an artifact path or includes weights.

    Cargo's slash-free include patterns match at any depth, even below a local
    symlink. Check its actual listing instead of assuming `.gitignore` prevails
    over an explicit package include. Compiler metadata warnings are not paths.
    Weight files are rejected except the tiny golden fixtures under ``root``
    that the packaged tests read (see ``is_tiny_golden_fixture``), and so is
    repository-only evidence: unpublished acceptance campaign directories and
    recorded mechanistic-study outputs (see ``package_contents.py``).
    """

    allowed_files = {"Cargo.toml", "Cargo.toml.orig", "Cargo.lock", "deny.toml",
                     "LICENSE-MIT", "LICENSE-APACHE", ".cargo_vcs_info.json"}
    allowed_roots = {"src", "examples", "tests", "scripts", "acceptance", "verus"}
    files = set()
    for raw in listing.splitlines():
        if not raw.strip() or raw.startswith("warning:") or raw.lstrip().startswith(("|", "= note:")):
            continue
        path = Path(raw)
        if path.is_absolute() or any(part in {"..", "__pycache__", "target", ".git"} for part in path.parts):
            raise ValueError(f"unsafe package entry: {raw}")
        if path.parts[:2] == ("scripts", "coverage-layout"):
            raise ValueError(f"repository-only standalone audit tool in runtime package: {raw}")
        if path.parts[0] == "verus" and (len(path.parts) != 2 or not (path.suffix == ".rs" or path.name == "README.md")):
            raise ValueError(f"unexpected entry below verus/: {raw}")
        if (path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth", ".bpk", ".gguf", ".pyc", ".pyo"}
                and not is_tiny_golden_fixture(path, root)):
            raise ValueError(f"runtime artifact in source package: {raw}")
        if ((path.parts[0] == "acceptance" and not package_contents.is_published_evidence(path))
                or package_contents.is_example_output(path)):
            raise ValueError(f"repository-only evidence in source package: {raw}")
        root_doc = len(path.parts) == 1 and path.suffix == ".md"
        if raw not in allowed_files and not root_doc and path.parts[0] not in allowed_roots:
            raise ValueError(f"unallowlisted package entry: {raw}")
        if raw in files:
            raise ValueError(f"duplicate package entry: {raw}")
        files.add(raw)
    if not {"Cargo.toml", "Cargo.lock", "README.md", "src/lib.rs"}.issubset(files):
        raise ValueError("incomplete Cargo package listing")
    return len(files)


def main() -> int:
    """Execute only requested gate groups and record failures without promotion."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="append", choices=GATES, required=True)
    parser.add_argument("--toolchain", default="stable")
    parser.add_argument("--msrv-toolchain", default="1.92.0")
    parser.add_argument("--offline", action="store_true", help="requires pre-fetched dependencies/advisories")
    parser.add_argument("--dry-run", action="store_true", help="print commands without creating files")
    parser.add_argument("--coverage-compiler-evidence", action="store_true",
                        help="require fresh pinned native compiler/export bindings; does not enable exclusions")
    args = parser.parse_args()
    if len(args.gate) != len(set(args.gate)):
        raise ValueError("duplicate gate groups would overwrite their evidence")
    if args.coverage_compiler_evidence and "coverage" not in args.gate:
        raise ValueError("--coverage-compiler-evidence requires the coverage gate")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = CRATE / "target/release-gates" / stamp
    plan = [(gate, command) for gate in args.gate for command in command_plan(
        gate, output, args.toolchain, args.msrv_toolchain, args.offline, verus=os.environ.get("VERUS"),
    )]
    last_verus = max((index for index, (gate, _) in enumerate(plan) if gate == "verus"), default=None)
    verus_state = None
    if args.coverage_compiler_evidence:
        for gate, command in plan:
            if gate == "coverage":
                command.append("--disable-default-ignore-filename-regex")
    if args.dry_run:
        for gate, command in plan:
            print(f"[{gate}] {shlex.join(command)}")
        return 0
    if "msrv" in args.gate:
        require_msrv(args.msrv_toolchain)
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.setdefault("CARGO_INCREMENTAL", "0")
    environment["CARGO_TERM_COLOR"] = "never"
    environment["RUSTDOCFLAGS"] = environment.get("RUSTDOCFLAGS", "") + " -D warnings"
    results = {"schema_version": 1, "created_utc": stamp, "gates": [], "passed": False}
    for index, (gate, command) in enumerate(plan):
        command_environment = dict(environment)
        log = output / f"{index:02d}-{gate}.log"
        print(f"[{gate}] {shlex.join(command)}\nlog: {log}", flush=True)
        started = datetime.now(timezone.utc)
        row = {"gate": gate, "command": command, "exit_code": 1,
               "process_exit_code": None, "log": log.name}
        results["gates"].append(row)
        try:
            if command_compiles(command):
                storage = row["storage_preflight"] = observe_build_storage(CRATE, command_environment)
                if not storage["passed"]:
                    reason = storage.get("measurement_error") or (
                        f"{storage['free_bytes']} bytes free; at least "
                        f"{storage['minimum_free_bytes']} bytes (20 GiB) required"
                    )
                    raise ValueError(f"compile storage preflight failed at {storage['checked_path']}: {reason}")
            if gate == "verus" and verus_state is None:
                verus_state = start_verus_evidence(CRATE)
            if gate == "guides":
                row["guide_doctests"] = write_guide_doctests(CRATE, output / "guide-doctests", args.offline)
            if gate == "coverage":
                if args.coverage_compiler_evidence:
                    command_environment, compiler_setup = prepare_capture(CRATE.resolve(), output.resolve(), args.toolchain, FEATURES, command_environment)
                    write_evidence(output / "compiler-capture-setup.json", compiler_setup)
                    row["compiler_evidence"] = {"setup": "compiler-capture-setup.json", "artifact_binding_verified": False,
                                                "compiler_reported_inputs_verified": False,
                                                "compiler_dependency_closure_verified": False,
                                                "production_only_exclusions_qualified": False}
                before = capture_coverage_context(CRATE, command, command_environment)
                before_path = output / "coverage-inputs-before.json"
                write_evidence(before_path, before)
                row["input_evidence"] = {"before": before_path.name,
                                         "sources_unchanged": False,
                                         "production_only_exclusions_qualified": False}
            with log.open("x") as stream:
                completed = subprocess.run(command, cwd=CRATE, env=command_environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
            row["process_exit_code"] = row["exit_code"] = completed.returncode
            if gate == "coverage":
                # Capture terminal observations even when the native process
                # fails. Failure never adopts counters or a passing manifest.
                after = capture_coverage_context(CRATE, command, command_environment)
                after_path = output / "coverage-inputs-after.json"
                write_evidence(after_path, after)
                row["input_evidence"]["after"] = after_path.name
                row["log_identity"] = fingerprint_file(log, maximum_bytes=512 * 1024 * 1024)
                report_path = output / "coverage.json"
                if report_path.exists() or report_path.is_symlink():
                    row["report_identity"] = fingerprint_file(report_path, maximum_bytes=2 * 1024 * 1024 * 1024)
                require_same_coverage_context(before, after)
                row["input_evidence"]["sources_unchanged"] = True
            if completed.returncode == 0 and gate == "coverage":
                if args.coverage_compiler_evidence:
                    binding = validate_capture(compiler_setup, before["sources"], CRATE.resolve())
                    notice_count = require_clean_coverage_log(read_coverage_log(log), allow_unfiltered_option_notice=(
                        binding["no_export_filter_verified"] is True and "--disable-default-ignore-filename-regex" in command))
                    binding["reviewed_tool_option_notice"] = {"text": UNFILTERED_OPTION_NOTICE, "count": notice_count}
                    native_report = read_json(Path(binding["native_report"]["path"]), 2 * 1024 * 1024 * 1024)
                    require_identity(Path(binding["native_report"]["path"]), binding["native_report"]["identity"])
                    require_bound_coverage_sources(native_report, before["sources"], CRATE)
                    # Bind raw native counters separately: cargo-llvm-cov's serde
                    # rewrite is not a byte-identical native export. Neither
                    # aggregate enables production-only source exclusions.
                    binding["native_aggregate"] = coverage_summary(native_report, CRATE)
                    binding["native_aggregate"]["scope"] = "unfiltered native export restricted to src/**/*.rs, including inline unit tests and generated expansions"
                    require_identity(Path(binding["native_report"]["path"]), binding["native_report"]["identity"])
                    write_evidence(output / "compiler-binding.json", binding)
                    row["compiler_evidence"].update({key: binding[key] for key in (
                        "artifact_binding_verified", "compiler_reported_inputs_verified", "no_export_filter_verified",
                    )})
                    row["compiler_evidence"]["binding"] = "compiler-binding.json"
                else:
                    require_clean_coverage_log(read_coverage_log(log))
                report = json.loads((output / "coverage.json").read_text())
                require_bound_coverage_sources(report, before["sources"], CRATE)
                row["coverage"] = coverage_summary(report, CRATE)
                # Detect ordinary edits during parsing as well as during the
                # native command. The hashes do not authenticate old binaries.
                if row["report_identity"] != fingerprint_file(report_path, maximum_bytes=2 * 1024 * 1024 * 1024):
                    raise ValueError("coverage export changed during validation")
                if row["log_identity"] != fingerprint_file(log, maximum_bytes=512 * 1024 * 1024):
                    raise ValueError("coverage log changed during validation")
                if not row["coverage"]["passed"]:
                    row["exit_code"] = 1
                if args.coverage_compiler_evidence and not binding["native_aggregate"]["passed"]:
                    row["exit_code"] = 1
            if completed.returncode == 0 and gate == "package" and "--list" in command:
                row["package_files"] = validate_package_listing(log.read_text())
            elif completed.returncode == 0 and gate == "package" and command[2:3] == ["package"]:
                row["package_archive"] = measure_package_archive(package_archive())
            if completed.returncode == 0 and gate == "verus":
                row.update(record_verus_step(verus_state, command, log.read_text()))
                if "harness" in row:
                    print(f"[verus] {row['harness']}: {row['verified']} verified, 0 errors", flush=True)
                if index == last_verus:
                    evidence = finish_verus_evidence(verus_state, CRATE)
                    write_evidence(output / "verus.json", evidence)
                    row["verus_evidence"] = "verus.json"
                    print(f"[verus] {len(evidence['harnesses'])} harnesses: {evidence['verified']} verified, 0 errors",
                          flush=True)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            # Preserve the distinction between a successful native process and
            # invalid evidence, including a malformed/missing export or unsafe
            # package inventory. Never leave an interrupted validation green.
            row["validation_error"] = str(error)
            row["exit_code"] = 1
            row.pop("coverage", None)
        row["seconds"] = (datetime.now(timezone.utc) - started).total_seconds()
        # A successfully executed prefix is not completion: an interruption or
        # missing tool at the next step must never leave a passing manifest.
        results["passed"] = index + 1 == len(plan) and all(item["exit_code"] == 0 for item in results["gates"])
        write_evidence(output / "results.json", results)
        if row["exit_code"] != 0:
            print(f"FAILED: {gate}; see {log} and {output / 'results.json'}", flush=True)
            return 1
    print(f"Requested gates passed; no release promotion performed. Evidence: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"release gate error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
