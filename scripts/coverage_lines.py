"""Reconstruct unique physical lines from LLVM 21 JSON coverage segments.

This is a measurement primitive, not a release gate or a source classifier. It
retains every supplied file, including tests, dependencies, and generated code.
LLVM's ``summary.lines`` instead aggregates function/instantiation-group counts,
so it is deliberately neither read nor used to validate physical-line totals.

The algorithm is pinned to llvmorg-21.1.8, ``LineCoverageStats`` and
``LineCoverageIterator`` in:
https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/lib/ProfileData/Coverage/CoverageMapping.cpp
Segment encoding is pinned to ``renderSegment`` in:
https://github.com/llvm/llvm-project/blob/llvmorg-21.1.8/llvm/tools/llvm-cov/CoverageExporterJson.cpp

Each segment is ``[line, column, count, has_count, region_entry, gap_region]``.
For a line, the *wrapped* segment is the final segment on an earlier line. A
non-gap counted region entry or a counted wrapped segment ordinarily maps the
line. A leading skipped entry suppresses the wrapped mapping; any counted entry
can restore it. Its execution count is the maximum of the wrapped count and
non-gap counted entries, not their sum. In particular, an end at column one can
still map that line through the wrapped segment, matching LLVM's line iterator.

Results are disjoint half-open line runs, avoiding allocation proportional to
large source coordinates. No source text, filesystem paths, profiles, or binary
objects are read. Callers must separately establish provenance and source scope.
"""

from collections.abc import Iterator
from typing import NamedTuple


LLVM_REVISION = "llvmorg-21.1.8"
EXPORT_TYPE = "llvm.coverage.json.export"
EXPORT_VERSION = "3.0.1"
MAX_COORDINATE = (1 << 32) - 1
MAX_COUNT = (1 << 63) - 1  # LLVM's JSON exporter clamps uint64 counters to int64.


class CoverageLinesError(ValueError):
    """An unsupported export or malformed segment cannot be measured safely."""


class LineRun(NamedTuple):
    """Mapped physical lines ``[start, stop)`` sharing one execution count."""

    start: int
    stop: int
    count: int


class FileLineCoverage(NamedTuple):
    """Normalized, non-overlapping mapped runs; zero-count runs remain present."""

    runs: tuple[LineRun, ...]

    @property
    def mapped_lines(self) -> int:
        """Count each mapped physical line once, regardless of region nesting."""

        return sum(run.stop - run.start for run in self.runs)

    @property
    def covered_lines(self) -> int:
        """Count mapped lines whose execution count is strictly positive."""

        return sum(run.stop - run.start for run in self.runs if run.count > 0)

    def iter_lines(self) -> Iterator[tuple[int, int]]:
        """Lazily expose ``(line, count)`` pairs without materializing long runs."""

        for run in self.runs:
            for line in range(run.start, run.stop):
                yield line, run.count


def _validate_segments(value: object, context: str) -> list[list]:
    """Validate the exporter schema and LLVM's flattened-segment invariants.

    Equal locations are legal only after an uncounted segment: LLVM emits this
    pair for a zero-length skipped region followed by its enclosing count.
    Counted segments at equal locations are ambiguous overlapping input, not
    additional independent evidence to sum. Nested source regions themselves
    have already been resolved by LLVM before segment export.
    """

    if not isinstance(value, list):
        raise CoverageLinesError(f"{context}: segments must be an array")
    previous = None
    for index, segment in enumerate(value):
        location = f"{context}: segment {index}"
        if not isinstance(segment, list) or len(segment) != 6:
            raise CoverageLinesError(f"{location}: expected six segment fields")
        line, column, count, has_count, entry, gap = segment
        for label, coordinate in (("line", line), ("column", column)):
            if type(coordinate) is not int or not 1 <= coordinate <= MAX_COORDINATE:
                raise CoverageLinesError(f"{location}: invalid {label} coordinate")
        if type(count) is not int or not 0 <= count <= MAX_COUNT:
            raise CoverageLinesError(f"{location}: invalid execution count")
        if any(type(flag) is not bool for flag in (has_count, entry, gap)):
            raise CoverageLinesError(f"{location}: segment flags must be booleans")
        if not has_count and (count != 0 or gap):
            raise CoverageLinesError(f"{location}: uncounted segment has a count or gap flag")
        if previous is not None:
            position, prior_position = (line, column), (previous[0], previous[1])
            if position < prior_position:
                raise CoverageLinesError(f"{location}: unsorted segment positions")
            if position == prior_position and previous[3]:
                raise CoverageLinesError(f"{location}: overlapping counted segment positions")
        previous = segment
    if previous is not None and previous[3]:
        raise CoverageLinesError(f"{context}: unterminated counted segment")
    return value


def _append_run(runs: list[LineRun], start: int, stop: int, count: int) -> None:
    """Coalesce adjacent equal-count runs without filling unmapped gaps."""

    if start == stop:
        return
    if runs and runs[-1].stop == start and runs[-1].count == count:
        runs[-1] = LineRun(runs[-1].start, stop, count)
    else:
        runs.append(LineRun(start, stop, count))


def line_coverage(segments: object, *, context: str = "<segments>") -> FileLineCoverage:
    """Apply LLVM 21 line semantics to one validated file's segment array.

    Runtime and storage are proportional to segment count, not the largest
    source line. Entire spans between segment-bearing lines have the wrapped
    count and can therefore be appended as one run. The final segment-bearing
    line is processed, but no line after it is inferred.
    """

    validated = _validate_segments(segments, context)
    runs: list[LineRun] = []
    wrapped = None
    index = 0
    previous_line = 0
    while index < len(validated):
        line = validated[index][0]
        if wrapped is not None and wrapped[3]:
            _append_run(runs, previous_line + 1, line, wrapped[2])
        first = index
        while index < len(validated) and validated[index][0] == line:
            index += 1
        current = validated[first:index]
        starts = [segment for segment in current if segment[3] and segment[4] and not segment[5]]
        skipped_start = not current[0][3] and current[0][4]
        mapped = (
            not skipped_start and (bool(starts) or wrapped is not None and wrapped[3])
        ) or any(segment[3] and segment[4] for segment in current)
        if mapped:
            count = max([wrapped[2] if wrapped is not None else 0] + [segment[2] for segment in starts])
            _append_run(runs, line, line + 1, count)
        wrapped = current[-1]
        previous_line = line
    return FileLineCoverage(tuple(runs))


def report_line_coverage(report: object) -> dict[str, FileLineCoverage]:
    """Measure every file in a supported decoded JSON export without exclusions.

    Require full segment exports, not summary-only reports. File names remain
    opaque strings and are never resolved or normalized. Duplicate names across
    data blocks fail: profiles must be merged by LLVM, not by this helper.
    Additional exporter metadata, summaries and function records are untouched.
    """

    if not isinstance(report, dict):
        raise CoverageLinesError("report: expected a JSON object")
    if report.get("type") != EXPORT_TYPE or report.get("version") != EXPORT_VERSION:
        raise CoverageLinesError(f"report: require {EXPORT_TYPE} version {EXPORT_VERSION}")
    data = report.get("data")
    if not isinstance(data, list) or not data:
        raise CoverageLinesError("report: data must be a nonempty array")
    measured = {}
    for block_index, block in enumerate(data):
        context = f"data[{block_index}]"
        if not isinstance(block, dict) or not isinstance(block.get("files"), list):
            raise CoverageLinesError(f"{context}: files must be an array")
        for file_index, entry in enumerate(block["files"]):
            location = f"{context}.files[{file_index}]"
            if not isinstance(entry, dict):
                raise CoverageLinesError(f"{location}: expected a file object")
            filename = entry.get("filename")
            if not isinstance(filename, str) or not filename or "\0" in filename:
                raise CoverageLinesError(f"{location}: filename must be a nonempty string without NUL")
            if filename in measured:
                raise CoverageLinesError(f"{location}: duplicate filename {filename!r}")
            if "segments" not in entry:
                raise CoverageLinesError(f"{location} {filename!r}: missing segments (summary-only export)")
            measured[filename] = line_coverage(entry["segments"], context=f"{location} {filename!r}")
    return measured
